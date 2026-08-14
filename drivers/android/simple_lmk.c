// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) 2019-2023 Sultan Alsawaf <sultan@kerneltoast.com>
 * Adapted for PSI event-driven triggers and self-contained configuration
 * on Linux 4.19 with Android PSI backport (includes psi_trigger_create()).
 *
 * Changes vs original polling-based adaptation:
 *  - Fixed PSI threshold scale (avg10 is a 0-100.00 percentage in this
 *    kernel's backport, NOT arbitrary large integers). Verify against
 *    `cat /proc/pressure/memory` before trusting these numbers on a
 *    different kernel/backport - the raw scale is not guaranteed
 *    identical across backports.
 *  - Tier selection is now driven by measured PSI severity (LOW/MED/HIGH)
 *    instead of always starting at the gentlest tier and only escalating
 *    reactively after a reclaim round fails to free enough memory.
 *  - Grace period for recently-active tasks now applies (with a shorter
 *    duration) across all tiers, not just the gentlest one, so apps
 *    backgrounded seconds ago aren't killed mid-transition under real
 *    pressure.
 *  - Swap-aware reclaim cap extended to tier 1 (moderate pressure), not
 *    just tier 0.
 *  - Kill logging rate-limited to avoid dmesg flooding during large
 *    kill storms.
 *  - Reclaim completion timeout raised slightly to give large mm's more
 *    time to fully unmap under RT priority.
 *  - PSI polling loop replaced with a single event-driven kernel-side
 *    PSI trigger (psi_trigger_create / wait_event on trigger->event_wait),
 *    eliminating up to 500ms detection latency and removing periodic
 *    wakeups entirely when the system is idle. Severity tiering is left
 *    to scan_and_kill()'s existing escalation logic on repeated calls,
 *    since a kthread cannot safely multiplex two separate PSI trigger
 *    waitqueues the way epoll-based userspace pollers can.
 *
 * UNTESTED: the PSI trigger integration has not been verified end-to-end
 * on real hardware as of this writing. Test in isolation before combining
 * with other changes. Keep the polling fallback (further below, disabled
 * by default) available in case the trigger path misbehaves on your
 * specific kernel's PSI backport.
 */

#define pr_fmt(fmt) "simple_lmk: " fmt

#include <linux/freezer.h>
#include <linux/jiffies.h>
#include <linux/kthread.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/oom.h>
#include <linux/sched/mm.h>
#include <linux/sort.h>
#include <linux/psi.h>
#include <linux/swap.h>
#include <linux/string.h>
#include <uapi/linux/sched/types.h>
#include <linux/delay.h>

/* config parameters */
#define LMK_MINFREE_MB                  128
#define LMK_TIMEOUT_MSEC                200
#define LMK_COOLDOWN_MSEC               1000

static unsigned long last_kill_completed_jiffies;
static unsigned int consecutive_fast_retriggers;


#define FAST_RETRIGGER_WINDOW  msecs_to_jiffies(3000)
#define FAST_RETRIGGER_BYPASS_THRESHOLD  2

#define SWAP_CRITICAL_FREE_PERCENT   15   /* if free swap drops below this
                                             percent of total swap, treat
                                             as an emergency regardless of
                                             computed RAM deficit */

/*
 * PSI avg10 thresholds, in the kernel's raw fixed-point scale.
 *
 * IMPORTANT: verify this scale on YOUR kernel before trusting these
 * numbers. Confirm with:
 *   pr_info("simple_lmk: raw_psi=%u\n", psi_system.avgs[PSI_MEM][0]);
 * compared against `cat /proc/pressure/memory`'s avg10= field at the
 * same moment. If e.g. avg10=18.33 corresponds to raw_psi=18330, the
 * scale is x1000 and the values below (representing 30.00%, 55.00%,
 * 75.00%) are correct. Adjust if your backport uses a different scale.
 */
#define LMK_PSI_THRESHOLD_LOW            5000   /* ~5.00% avg10 */
#define LMK_PSI_THRESHOLD_MED            10000   /* ~10.00% avg10 */
#define LMK_PSI_THRESHOLD_HIGH           15000   /* ~15.00% avg10 */

/* PSI trigger window/threshold, in microseconds (separate scale from
 * the avg10 percentage above - this is raw stall time within a window,
 * consumed only by psi_trigger_create(), not compared against avg10). */
#define LMK_TRIGGER_STALL_US             270000  /* 270ms stall */
#define LMK_TRIGGER_WINDOW_US            1100000 /* within 1s window */

/* Set to 1 to use the legacy polling loop instead of the PSI trigger.
 * Useful as a fallback if psi_trigger_create() misbehaves on your
 * specific kernel/backport. */
#define LMK_USE_POLLING_FALLBACK         0
#define LMK_POLL_INTERVAL_IDLE_MSEC      500
#define LMK_POLL_INTERVAL_BUSY_MSEC      100
/* ------------------------------------------------------------- */

/* Grace period in milliseconds for newly backgrounded apps.
 * Applied at full duration in the gentlest tier, and at a reduced
 * duration in tier 1 (moderate pressure) so apps that were foreground
 * seconds ago aren't killed mid-transition. Tier 2 (severe pressure)
 * ignores grace period entirely - correctness/stability takes priority
 * over avoiding a cold start at that point. */
#define GRACE_PERIOD_MS_TIER0 5000
#define GRACE_PERIOD_MS_TIER1 1500

/* Kill up to this many victims per reclaim */
#define MAX_VICTIMS 102

#define MIN_FREE_PAGES (LMK_MINFREE_MB * SZ_1M / PAGE_SIZE)

#define OOM_ADJ_MIN (-1024)
#define OOM_ADJ_MAX 1024
#define OOM_ADJ_BUCKET_COUNT (OOM_ADJ_MAX - OOM_ADJ_MIN + 1)

#define adj_to_bucket(adj) ((adj) - OOM_ADJ_MIN)

#define RECLAIM_EXPIRES msecs_to_jiffies(LMK_TIMEOUT_MSEC)
#define RECLAIM_COOLDOWN msecs_to_jiffies(LMK_COOLDOWN_MSEC)

#define LMK_TIERS 3
static const short tier_min_adj[LMK_TIERS] = { 600, 200, 1 };
/* Swap-aware reclaim cap (in bytes worth of pages) applied per tier when
 * swap has headroom. Only tiers 0 and 1 respect this; tier 2 (severe)
 * always reclaims the full computed deficit regardless of swap state. */
static const unsigned long tier_swap_cap_mb[LMK_TIERS] = { 32, 64, 0 };

struct victim_info {
	struct task_struct *tsk;
	struct mm_struct *mm;
	unsigned long size;
};

static struct victim_info victims[MAX_VICTIMS] __cacheline_aligned_in_smp;
static struct task_struct *task_bucket[OOM_ADJ_BUCKET_COUNT] __cacheline_aligned;
static DECLARE_WAIT_QUEUE_HEAD(oom_waitq);
static DECLARE_WAIT_QUEUE_HEAD(reaper_waitq);
static DECLARE_COMPLETION(reclaim_done);
static __cacheline_aligned_in_smp DEFINE_RWLOCK(mm_free_lock);
static int nr_victims;
static bool reclaim_active;
static atomic_t needs_reclaim = ATOMIC_INIT(0);
static atomic_t needs_reap = ATOMIC_INIT(0);
static atomic_t nr_killed = ATOMIC_INIT(0);
static atomic_t target_min_adj = ATOMIC_INIT(tier_min_adj[0]);

static unsigned long last_reclaim_jiffies;

static int tier_index(short min_adj)
{
	int i;

	for (i = 0; i < LMK_TIERS; i++)
		if (tier_min_adj[i] == min_adj)
			return i;
	return 0;
}


static int victim_cmp(const void *lhs_ptr, const void *rhs_ptr)
{
	const struct victim_info *lhs = (typeof(lhs))lhs_ptr;
	const struct victim_info *rhs = (typeof(rhs))rhs_ptr;

	if (rhs->size > lhs->size)
		return 1;
	if (rhs->size < lhs->size)
		return -1;
	return 0;
}

static void victim_swap(void *lhs_ptr, void *rhs_ptr, int size)
{
	struct victim_info *lhs = (typeof(lhs))lhs_ptr;
	struct victim_info *rhs = (typeof(rhs))rhs_ptr;

	swap(*lhs, *rhs);
}

static unsigned long get_total_mm_pages(struct mm_struct *mm)
{
	unsigned long pages = 0;
	int i;

	for (i = 0; i < NR_MM_COUNTERS; i++)
		pages += get_mm_counter(mm, i);

	return pages;
}

static void release_stale_victims(void)
{
	int i, old_nr;

	write_lock(&mm_free_lock);
	old_nr = nr_victims;
	nr_victims = 0;
	write_unlock(&mm_free_lock);

	for (i = 0; i < old_nr; i++) {
		if (victims[i].mm) {
			mmdrop(victims[i].mm);
			victims[i].mm = NULL;
		}
	}
}

unsigned long get_target_free_pages(short limit_adj)
{
	unsigned long deficit;
	unsigned long swap_cap_mb;
	struct sysinfo val;
	int tidx = tier_index(limit_adj);

    si_swapinfo(&val);

    if (val.totalswap > 0) {
		unsigned long swap_free_pct =
			(val.freeswap * 100) / val.totalswap;

		if (swap_free_pct < SWAP_CRITICAL_FREE_PERCENT) {
			pr_err_ratelimited(
				"SWAP CRITICAL: %lu%% free, forcing aggressive reclaim (target=%luMB)\n",
        swap_free_pct, (256UL * SZ_1M / PAGE_SIZE) * PAGE_SIZE / SZ_1M);
			/* Force tier 2 immediately regardless of what tier
			 * we were asked for, and reclaim a large fixed
			 * chunk unconditionally - don't rely on the RAM
			 * reserve deficit calculation at all here, since
			 * that's precisely what failed to catch this. */
			atomic_set(&target_min_adj, tier_min_adj[2]);
			return (256 * SZ_1M) / PAGE_SIZE; /* 256MB, tune as needed */
		}
	}

	if (nr_free_pages() >= totalreserve_pages)
		return 0;

	deficit = totalreserve_pages - nr_free_pages();
	deficit += (deficit >> 3); /* 12.5% margin */

	/* If we're re-triggering rapidly despite recent kills, the cap
	 * is clearly insufficient for current demand - bypass it and
	 * reclaim the full deficit instead of death-spiraling at ~1Hz. */
	if (consecutive_fast_retriggers >= FAST_RETRIGGER_BYPASS_THRESHOLD)
		return deficit;

	swap_cap_mb = tier_swap_cap_mb[tidx];
	if (swap_cap_mb) {
		if (val.freeswap > (totalram_pages >> 3))
			return min_t(unsigned long, deficit,
				     swap_cap_mb * SZ_1M / PAGE_SIZE);
	}

	return deficit;
}


static unsigned long find_victims(int *vindex)
{
	short i, limit_adj = atomic_read(&target_min_adj);
	short min_adj = SHRT_MAX, max_adj = 0;
	unsigned long pages_found = 0;
	unsigned long target_pages = get_target_free_pages(limit_adj);
	unsigned long grace_ms;
	struct task_struct *tsk;

	if (target_pages == 0)
		return 0;

	/* Grace period: full duration in tier 0, reduced in tier 1,
	 * none in tier 2 (severe pressure). */
	if (limit_adj == tier_min_adj[0])
		grace_ms = GRACE_PERIOD_MS_TIER0;
	else if (limit_adj == tier_min_adj[1])
		grace_ms = GRACE_PERIOD_MS_TIER1;
	else
		grace_ms = 0;

	rcu_read_lock();
	for_each_process(tsk) {
		struct signal_struct *sig;
		short adj;

		sig = tsk->signal;
		adj = READ_ONCE(sig->oom_score_adj);
		if (adj < 0 ||
		    sig->flags & (SIGNAL_GROUP_EXIT | SIGNAL_GROUP_COREDUMP) ||
		    (thread_group_empty(tsk) && tsk->flags & PF_EXITING))
			continue;

		if (adj > OOM_ADJ_MAX)
			adj = OOM_ADJ_MAX;

		/* Track the last time this task was seen below the cached
		 * tier threshold (i.e. foreground/perceptible-ish), as an
		 * approximation for grace-period purposes. Updated on every
		 * scan pass instead of via a tracepoint hook, since no
		 * oom_score_adj_update tracepoint exists in this kernel. */
		if (adj < tier_min_adj[0])
			tsk->simple_lmk_cache_time = jiffies;

		tsk->simple_lmk_next = task_bucket[adj_to_bucket(adj)];
		task_bucket[adj_to_bucket(adj)] = tsk;

		if (adj > max_adj)
			max_adj = adj;
		if (adj < min_adj)
			min_adj = adj;
	}


	for (i = max_adj; i >= limit_adj; i--) {
		int old_vindex;
		int bucket_idx;

		if (i < min_adj || i > max_adj)
			continue;

		bucket_idx = adj_to_bucket(i);
		tsk = task_bucket[bucket_idx];
		if (!tsk)
			continue;

		task_bucket[bucket_idx] = NULL;
		old_vindex = *vindex;
		do {
			struct task_struct *vtsk;

			if (grace_ms && i >= tier_min_adj[0] &&
			    time_before(jiffies, tsk->simple_lmk_cache_time +
					msecs_to_jiffies(grace_ms)))
				continue;

			vtsk = find_lock_task_mm(tsk);
			if (!vtsk)
				continue;

			mmgrab(vtsk->mm);

			victims[*vindex].tsk = vtsk;
			victims[*vindex].mm = vtsk->mm;
			victims[*vindex].size = get_total_mm_pages(vtsk->mm);

			pages_found += victims[*vindex].size;

			if (++*vindex == MAX_VICTIMS || pages_found >= target_pages)
				break;
		} while ((tsk = tsk->simple_lmk_next));

		if (*vindex == old_vindex)
			continue;

		sort(&victims[old_vindex], *vindex - old_vindex,
		     sizeof(*victims), victim_cmp, victim_swap);

		if (*vindex == MAX_VICTIMS || pages_found >= target_pages)
			break;
	}

	

	if (min_adj <= max_adj) {
		int start = adj_to_bucket(min_adj);
		int end = adj_to_bucket(max_adj);

		memset(&task_bucket[start], 0,
		       (end - start + 1) * sizeof(*task_bucket));
	}

	rcu_read_unlock();

	return pages_found;
}


static int process_victims(int vlen)
{
	unsigned long pages_found = 0;
	short limit_adj = atomic_read(&target_min_adj);
	unsigned long target_pages = get_target_free_pages(limit_adj);
	int i, nr_to_kill = 0;

	for (i = 0; i < vlen; i++) {
		struct victim_info *victim = &victims[i];
		struct task_struct *vtsk = victim->tsk;

		if (pages_found >= target_pages) {
			task_unlock(vtsk);
			mmdrop(victim->mm);
			victim->mm = NULL;
		} else {
			pages_found += victim->size;
			nr_to_kill++;
		}
	}

	return nr_to_kill;
}

static void set_task_rt_prio(struct task_struct *tsk, int priority)
{
	const struct sched_param rt_prio = {
		.sched_priority = priority
	};

	sched_setscheduler_nocheck(tsk, SCHED_RR, &rt_prio);
}

static void scan_and_kill(void)
{
	int i, nr_to_kill, nr_found = 0;
	unsigned long pages_found;
	unsigned long total_freed_kib = 0;
	int current_tier;
	int next_adj;

	release_stale_victims();

	/* If there's no actual memory deficit right now, don't bother
	 * scanning at all - a deficit of 0 doesn't change across tiers. */
	if (get_target_free_pages(atomic_read(&target_min_adj)) == 0)
		return;

	for (current_tier = 0; current_tier < LMK_TIERS; current_tier++) {
		pages_found = find_victims(&nr_found);
		if (nr_found > 0)
			break;

		next_adj = tier_min_adj[current_tier + 1 < LMK_TIERS ? current_tier + 1 : LMK_TIERS - 1];
		atomic_set(&target_min_adj, next_adj);
	}
	

	if (unlikely(!nr_found)) {
		pr_err_ratelimited("No processes available to kill even across all tiers!\n");
		return;
	}

	nr_to_kill = process_victims(nr_found);
	sort(victims, nr_to_kill, sizeof(*victims), victim_cmp, victim_swap);
	nr_to_kill = process_victims(nr_to_kill);
    

	if (nr_to_kill <= 0) {
		write_lock(&mm_free_lock);
		reclaim_active = false;
		atomic_set(&nr_killed, 0);
		reinit_completion(&reclaim_done);
		write_unlock(&mm_free_lock);
		return;
	}

	for (i = 0; i < nr_to_kill; i++) {
		struct victim_info *victim = &victims[i];
		struct task_struct *t, *vtsk = victim->tsk;
		struct mm_struct *mm = victim->mm;
		unsigned long victim_kib = victim->size << (PAGE_SHIFT - 10);

		pr_info_ratelimited("Killing %s with adj %d to free %lu KiB\n",
			vtsk->comm, vtsk->signal->oom_score_adj, victim_kib);
		total_freed_kib += victim_kib;

		set_bit(MMF_OOM_VICTIM, &mm->flags);
		do_send_sig_info(SIGKILL, SEND_SIG_FORCED, vtsk, PIDTYPE_TGID);

		rcu_read_lock();
		for_each_thread(vtsk, t)
			set_tsk_thread_flag(t, TIF_MEMDIE);
		for_each_thread(vtsk, t)
			set_task_rt_prio(t, 1);
		rcu_read_unlock();

		set_cpus_allowed_ptr(vtsk, cpu_all_mask);
		__thaw_task(vtsk);

		victim->size = get_mm_counter(mm, MM_ANONPAGES);
		task_unlock(vtsk);
	}

	pr_info("Killed %d process(es), freed ~%lu KiB total (nr_found=%d)\n",
    nr_to_kill, total_freed_kib, nr_found);

	write_lock(&mm_free_lock);
	sort(victims, nr_to_kill, sizeof(*victims), victim_cmp, victim_swap);
	atomic_set(&needs_reap, 1);
	write_unlock(&mm_free_lock);
	wake_up(&reaper_waitq);

	if (!wait_for_completion_timeout(&reclaim_done, RECLAIM_EXPIRES))
		pr_info_ratelimited("Timeout hit waiting for victims to die, proceeding\n");

    /* Track whether this trigger fired shortly after the previous
	 * kill completed - repeated fast retriggers mean our reclaim
	 * amount is insufficient for current demand. */
	if (last_kill_completed_jiffies &&
	    time_before(jiffies, last_kill_completed_jiffies + FAST_RETRIGGER_WINDOW))
		consecutive_fast_retriggers++;
	else
		consecutive_fast_retriggers = 0;

	last_kill_completed_jiffies = jiffies;

	if (nr_free_pages() >= totalreserve_pages) {
		atomic_set(&target_min_adj, tier_min_adj[0]);
	} else {
		int current_adj = atomic_read(&target_min_adj);
		if (current_adj == tier_min_adj[0])
			atomic_set(&target_min_adj, tier_min_adj[1]);
		else if (current_adj == tier_min_adj[1])
			atomic_set(&target_min_adj, tier_min_adj[2]);
	}

	write_lock(&mm_free_lock);
	reclaim_active = false;
	atomic_set(&nr_killed, 0);
	reinit_completion(&reclaim_done);
	write_unlock(&mm_free_lock);
}

static int simple_lmk_reclaim_thread(void *data)
{
	set_task_rt_prio(current, MAX_RT_PRIO - 1);
	set_freezable();

	while (1) {
		wait_event_freezable(oom_waitq,
				     atomic_read_acquire(&needs_reclaim));
		scan_and_kill();
		atomic_set_release(&needs_reclaim, 0);
	}

	return 0;
}

static struct mm_struct *next_reap_victim(void)
{
	struct mm_struct *mm = NULL;
	bool should_retry = false;
	int i;

	write_lock(&mm_free_lock);
	for (i = 0; i < nr_victims; i++, mm = NULL) {
		mm = victims[i].mm;
		if (!mm || test_bit(MMF_OOM_SKIP, &mm->flags))
			continue;

		if (!down_read_trylock(&mm->mmap_sem)) {
			should_retry = true;
			continue;
		}

		if (!test_bit(MMF_OOM_SKIP, &mm->flags))
			break;
		up_read(&mm->mmap_sem);
	}

	if (!mm) {
		if (should_retry) {
			mm = ERR_PTR(-EAGAIN);
		} else if (!reclaim_active) {
			nr_victims = 0;
		}
	}
	write_unlock(&mm_free_lock);

	return mm;
}

static void reap_victims(void)
{
	struct mm_struct *mm;

	while ((mm = next_reap_victim())) {
		if (IS_ERR(mm)) {
			schedule_timeout_uninterruptible(1);
			continue;
		}

		if (__oom_reap_task_mm(mm)) {
			clear_bit(MMF_OOM_VICTIM, &mm->flags);
			set_bit(MMF_OOM_SKIP, &mm->flags);
		}
		up_read(&mm->mmap_sem);
		
		cond_resched(); 
	}
}

static int simple_lmk_reaper_thread(void *data)
{
	set_task_rt_prio(current, MAX_RT_PRIO - 2);
	set_freezable();

	while (1) {
		wait_event_freezable(reaper_waitq,
				     atomic_cmpxchg(&needs_reap, 1, 0));
		reap_victims();
	}

	return 0;
}

void simple_lmk_mm_freed(struct mm_struct *mm)
{
	int i;
	bool do_mmdrop = false;

	if (!test_bit(MMF_OOM_SKIP, &mm->flags))
		return;

	read_lock(&mm_free_lock);
	for (i = 0; i < nr_victims; i++) {
		if (victims[i].mm == mm) {
			victims[i].mm = NULL;
			do_mmdrop = true;
			if (reclaim_active &&
			    atomic_inc_return_relaxed(&nr_killed) == nr_victims)
				complete(&reclaim_done);
			break;
		}
	}
	read_unlock(&mm_free_lock);

	if (do_mmdrop)
		mmdrop(mm);
}

#if LMK_USE_POLLING_FALLBACK
/*
 * Legacy polling implementation, kept as a fallback. Enable by setting
 * LMK_USE_POLLING_FALLBACK to 1 above if the PSI trigger path below
 * causes issues on your kernel. Uses corrected thresholds and adaptive
 * poll interval (polls faster while already under some pressure).
 */
static int simple_lmk_psi_thread(void *data)
{
    int log_counter = 0;
    pr_err("PSI thread ENTRY POINT reached!\n");
	set_task_rt_prio(current, MAX_RT_PRIO - 3);
	set_freezable();

    pr_err("PSI monitoring thread started. Low threshold: %d\n", LMK_PSI_THRESHOLD_LOW);

	while (!kthread_should_stop()) {
		unsigned int avg10_val = 0;

		#ifdef CONFIG_PSI
		avg10_val = psi_system.avg[PSI_MEM_SOME][0];
		#endif

        if (++log_counter >= 20) {
			pr_err("Current memory PSI avg10 = %u (Threshold: %d)\n", avg10_val, LMK_PSI_THRESHOLD_LOW);
			log_counter = 0;
		}

		if (avg10_val >= LMK_PSI_THRESHOLD_LOW &&
		    time_after_eq(jiffies, last_reclaim_jiffies + RECLAIM_COOLDOWN)) {
			short tier;

			if (avg10_val >= LMK_PSI_THRESHOLD_HIGH)
				tier = tier_min_adj[2];
			else if (avg10_val >= LMK_PSI_THRESHOLD_MED)
				tier = tier_min_adj[1];
			else
				tier = tier_min_adj[0];

            pr_err("Pressure threshold reached! avg10=%u, escalating to tier adj %d\n", avg10_val, tier);

			/* Only escalate to a more aggressive tier here; let
			 * scan_and_kill() handle de-escalation on success. */
			if (tier < atomic_read(&target_min_adj))
				atomic_set(&target_min_adj, tier);

			last_reclaim_jiffies = jiffies;
			atomic_set_release(&needs_reclaim, 1);
			wake_up(&oom_waitq);
		}

		msleep(avg10_val >= LMK_PSI_THRESHOLD_LOW ?
		       LMK_POLL_INTERVAL_BUSY_MSEC : LMK_POLL_INTERVAL_IDLE_MSEC);
	}

	return 0;
}
#else
/*
 * Event-driven PSI trigger implementation (default). Uses a single
 * kernel-side PSI trigger and blocks on its waitqueue instead of
 * polling, eliminating detection latency and idle wakeups.
 *
 * Severity tiering is left to scan_and_kill()'s existing escalation
 * logic across repeated calls, since safely multiplexing multiple
 * separate PSI trigger waitqueues from a single kthread is not
 * straightforward (psi_trigger_poll() is designed for epoll-style
 * multi-fd waiting from userspace, not direct kthread use).
 *
 * UNTESTED on real hardware as of this writing - verify behavior
 * under actual memory pressure before relying on this in production.
 */
static struct psi_trigger *lmk_trigger;

static int simple_lmk_psi_thread(void *data)
{
	char trig_buf[64];

	set_task_rt_prio(current, MAX_RT_PRIO - 3);
	set_freezable();

	snprintf(trig_buf, sizeof(trig_buf), "some %u %u\n",
		 LMK_TRIGGER_STALL_US, LMK_TRIGGER_WINDOW_US);

	lmk_trigger = psi_trigger_create(&psi_system, trig_buf,
					  strlen(trig_buf), PSI_MEM);
	if (IS_ERR(lmk_trigger)) {
		pr_err("PSI trigger init failed: %ld\n", PTR_ERR(lmk_trigger));
		lmk_trigger = NULL;
		return PTR_ERR(lmk_trigger);
	}
    pr_err("PSI trigger initialized successfully\n");

	while (!kthread_should_stop()) {
    wait_event_freezable(lmk_trigger->event_wait,
        cmpxchg(&lmk_trigger->event, 1, 0) == 1 ||
        kthread_should_stop());

    if (kthread_should_stop())
        break;

    /* Bypass cooldown entirely if swap is critically low - waiting
     * a full second between reclaim attempts during an active
     * emergency is how this turns into a full freeze. */
    {
        struct sysinfo val;
        si_swapinfo(&val);
        if (val.totalswap > 0 &&
            (val.freeswap * 100 / val.totalswap) < SWAP_CRITICAL_FREE_PERCENT) {
            atomic_set_release(&needs_reclaim, 1);
            wake_up(&oom_waitq);
            continue;
        }
    }

    if (!time_after_eq(jiffies, last_reclaim_jiffies + RECLAIM_COOLDOWN))
        continue;

    last_reclaim_jiffies = jiffies;
    atomic_set_release(&needs_reclaim, 1);
    wake_up(&oom_waitq);
}

	if (lmk_trigger) {
		psi_trigger_destroy(lmk_trigger);
		lmk_trigger = NULL;
	}

	return 0;
}
#endif /* LMK_USE_POLLING_FALLBACK */

static int simple_lmk_init_set(const char *val, const struct kernel_param *kp)
{
	static atomic_t init_done = ATOMIC_INIT(0);
	struct task_struct *thread;
	int ret;

	if (!atomic_cmpxchg(&init_done, 0, 1)) {
        pr_err("Driver initialized successfully! (Threshold LOW: %d)\n", LMK_PSI_THRESHOLD_LOW);
		thread = kthread_run(simple_lmk_reaper_thread, NULL,
				     "simple_lmkd_reaper");
		if (WARN_ON(IS_ERR(thread))) {
			ret = PTR_ERR(thread);
			goto err_reset;
		}

		thread = kthread_run(simple_lmk_reclaim_thread, NULL,
				     "simple_lmkd");
		if (WARN_ON(IS_ERR(thread))) {
			ret = PTR_ERR(thread);
			goto err_reset;
		}

		thread = kthread_run(simple_lmk_psi_thread, NULL,
				     "simple_lmkd_psi");
		if (WARN_ON(IS_ERR(thread))) {
			ret = PTR_ERR(thread);
			goto err_reset;
		}
	}

	return 0;

err_reset:
	atomic_set(&init_done, 0);
	return ret;
}

static const struct kernel_param_ops simple_lmk_init_ops = {
	.set = simple_lmk_init_set
};

#undef MODULE_PARAM_PREFIX
#define MODULE_PARAM_PREFIX "lowmemorykiller."
module_param_cb(minfree, &simple_lmk_init_ops, NULL, 0200);

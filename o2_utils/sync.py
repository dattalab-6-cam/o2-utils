import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.stats import linregress


def _get_aligned_times_based_on_diff_correlation(
    source_a_times, source_b_times, make_plots=False
):
    """Cross-correlate the times between matching sync events to align them.
    NB, if the events do not match perfectly, this function will come as close as possible to aligning them.

    Parameters
    ----------
    source_a_times : np.array, shape (n,)
        The timestamps of events in the first source timebase.

    source_b_times : np.array, shape (n,)
        The timestamps of events in the second source timebase.

    make_plots : bool, optional
        Whether to make plots of the cross-correlation, by default False.

    Returns
    -------
    np.array, np.array
        The aligned timestamps for the first and second source time, based on peak correlation.

    """
    # get diffs
    diff_a = np.diff(source_a_times)
    diff_b = np.diff(source_b_times)

    cc = correlate(diff_a - diff_a.mean(), diff_b - diff_b.mean(), mode="full")
    # cc /= np.std(diff_a) * np.std(diff_b) * len(diff_a)
    lags = correlation_lags(len(diff_a), len(diff_b), mode="full")
    max_idx = np.argmax(cc)
    peak_lag = lags[max_idx]
    # print(f"Max correlation: {cc[max_idx]}")

    if make_plots:
        plt.figure()
        plt.plot(lags, cc)

    # get back aligned times lined up 1:1, ie same num timestamps in each
    min_len = np.min([len(source_a_times), len(source_b_times)])
    if peak_lag < 0:
        b_use = source_b_times[(-1 * peak_lag) : (min_len - peak_lag)]
        a_use = source_a_times[: len(b_use)]
    elif peak_lag > 0:
        a_use = source_a_times[peak_lag : (min_len + peak_lag)]
        b_use = source_b_times[: len(a_use)]
    else:
        a_use = source_a_times[:min_len]
        b_use = source_b_times[:min_len]

    # verify
    assert len(a_use) == len(b_use)

    if make_plots:
        plt.figure()
        if min_len > 250:
            s = slice(0, 250)
        else:
            s = slice(0, len(a_use))
        plt.plot(np.diff(a_use[s]))
        plt.plot(np.diff(b_use[s]))

    return a_use, b_use


def _convert_timestamps(dest_event_times, source_event_times, full_source_timestamps):
    """Convert timestamps from a source timebase to a destination timebase with a linear model
    based on aligned events across the two timebases.

    NB: this function assumes the source and destination timebases are already aligned!

    Parameters
    ----------
    dest_event_times : np.array, shape (n,)
        The timestamps of the destination timebase. (Treated as ground truth, ie X, in the model).

    source_event_times : np.array, shape (n,)
        The timestamps of the source timebase. (Treated as the recorded times, ie Y, in the model)

    full_source_timestamps : np.array
        The full set of timestamps from the source timebase which will be converted to the destination timebase.

    Returns
    -------
    np.array
        The timestamps from the source timebase converted to the destination timebase.
    """
    # Make sure the sync signals are aligned.
    assert len(dest_event_times) == len(source_event_times)
    
    # This assert will fail in the case of partitioned recordings
    # where the timestamps can sometimes have large jumps from teh
    # event at the end of one partition to the event at the start of the next.
    # As long as we've checked this in each partition, we're good.
    # assert (
    #     np.sum(
    #         ~np.isclose(
    #             np.diff(dest_event_times), np.diff(source_event_times), atol=3e-3
    #         )
    #     )
    #     == 0
    # )

    # Make a model to convert between the two timebases
    mdl = linregress(
        dest_event_times, source_event_times
    )  # treat dest_times times as gnd truth, error is in recorded times by some DAC
    f = (
        lambda x: (x - mdl.intercept) / mdl.slope
    )  # linregress is better than interpolation here.
    print(f"mdl slope: {mdl.slope}, mdl intercept: {mdl.intercept}")
    tfic = f(full_source_timestamps)
    return tfic, f


def check_aligned_events(dest_event_times, source_event_times, atol=3e-3):
    len_check = len(dest_event_times) == len(source_event_times)
    diff_check = (
        np.sum(
            ~np.isclose(
                np.diff(dest_event_times), np.diff(source_event_times), atol=atol
            )
        )
        == 0
    )
    return len_check and diff_check


def align_sync_signals(
    dest_event_times,
    source_event_times,
    full_source_timestamps,
    make_plots=False,
    partition_fraction_on_error=4,
):
    """Error-robust method to align two sync signals based on cross-correlation of the inter-event intervals.

    This error-robust method allows for missing events in either signal. The logic is:
    1. try to perform normal alignment with cross-correlation.
    2. check that all inter-event intervals match. If so, done. If not, split the event vectors in halves/quarters/etc
        to isolate the issue, and then return aligned timestamps excluding the problematic events.
    3. Use aligned timestamps to fit a model to convert between the two timebases.

    Parameters
    ----------
    dest_event_times : np.array, shape (n,)
        Event timestamps in the destination timebase. (Treated as ground truth, ie X, in the model).

    source_event_times : np.array, shape (n,)
        Event timestamps in the source timebase. (Treated as the recorded times, ie Y, in the model)

    full_source_timestamps : np.array
        The full set of timestamps from the source timebase which will be converted to the destination timebase.

    make_plots : bool, optional
        Whether to make plots of the cross-correlation, by default False.

    partition_fraction_on_error : int, optional
        The number of partitions to split the event vectors into if the inter-event intervals do not match, by default 4.

    Returns
    -------
    tfic: np.array
        The timestamps from the source timebase converted to the destination timebase, and the linear model to convert between the two timebases.

    conversion_func: function
        The linear model to convert between the two timebases (ie f(source_time) -> dest_time).
    """

    # Try initial correlation
    (
        dest_aligned_times,
        source_aligned_times,
    ) = _get_aligned_times_based_on_diff_correlation(
        dest_event_times, source_event_times, make_plots=make_plots
    )

    # Check that all inter-event intervals match. If not, split the event vectors in halves/quarters/etc.
    if check_aligned_events(dest_aligned_times, source_aligned_times):
        print("All inter-event intervals match.")
    else:
        print(
            "Inter-event intervals do not match. Partitioning into sub-recordings to isolate issue..."
        )
        dest_event_times_partitioned = np.array_split(
            dest_aligned_times, partition_fraction_on_error
        )
        source_event_times_partitioned = np.array_split(
            source_aligned_times, partition_fraction_on_error
        )
        aligned_partitioned_times = {"dest": [], "source": [], "matches": []}
        for i, (dest_partition, source_partition) in enumerate(
            zip(dest_event_times_partitioned, source_event_times_partitioned)
        ):
            (
                dest_part_aligned,
                source_part_aligned,
            ) = _get_aligned_times_based_on_diff_correlation(
                dest_partition, source_partition, make_plots=make_plots
            )
            if check_aligned_events(dest_part_aligned, source_part_aligned):
                aligned_partitioned_times["dest"].append(dest_part_aligned)
                aligned_partitioned_times["source"].append(source_part_aligned)
                aligned_partitioned_times["matches"].append(i)
                print(f"Partition {i} inter-event intervals match.")
            else:
                print(f"Partition {i} inter-event intervals do NOT match.")

        if len(aligned_partitioned_times["matches"]) == 0:
            raise ValueError("No partitions had matching inter-event intervals.")

        # Concatenate the aligned partitions
        dest_aligned_times = np.concatenate(aligned_partitioned_times["dest"])
        source_aligned_times = np.concatenate(aligned_partitioned_times["source"])
        print(len(dest_aligned_times), len(source_aligned_times))
        assert len(dest_aligned_times) == len(source_aligned_times)

    # Convert timestamps from source timebase to destination timebase
    tfic, conversion_func = _convert_timestamps(
        dest_aligned_times, source_aligned_times, full_source_timestamps
    )

    return tfic, conversion_func

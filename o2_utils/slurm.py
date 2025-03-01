import os
import re
from copy import copy

import matplotlib.pyplot as plt
import numpy as np


def parse_sacct_line(line, line_type):
    """Parse a line of output from the O2sacct command

    Parameters
    ----------
    line : str
        Line of output from the O2sacct command
    line_type : str
        Either "batch" or "job", depending on which type of line is being parsed

    Returns
    -------
    job_data : dict
        Dictionary containing information about the job
    """

    # Split line based on multiple spaces
    parts = line.split()
    
    # Assuming the columns are always in the same order as the sample provided
    cpu_regexp = re.compile(r"cpu=(?P<cpu>\d+)")
    mem_regexp = re.compile(r"mem=(?P<mem>\d+[A-Z])")
    if line_type == "batch":
        job_data = {
            'JobID': parts[0].split(".")[0],
            "Status": parts[1],
            'NodeList': parts[2],
            'StartTime': parts[3],
            'ElapsedTime': parts[4], 
            'CPUTimeRaw': parts[5], 
            'N_CPUs': cpu_regexp.search(parts[6]).groups()[0],
            'Mem_Req': mem_regexp.search(parts[6]).groups()[0],
            'Mem_Used': parts[7] if ((parts[1] != "RUNNING") and (parts[1] != "PENDING")) else np.nan,
        }
    elif line_type == "job":
        job_data = {
            'JobID': parts[0],
            "Partition": parts[1],
            'RequestedTime': parts[5], 
        }
    return job_data


def get_job_info(jobid):
    """Use the O2sacct command to get information about a single job

    Parameters
    ----------
    jobid : str
        Jobid to get information about

    Returns
    -------
    batch_info : dict
        Dictionary containing information about the batch job

    Example output:
        {'JobID': '32828121',
        'Status': 'COMPLETED',
        'NodeList': 'compute-g-17-163',
        'StartTime': '2024-02-23T23:27:34',
        'ElapsedTime': '00:45:40',
        'CPUTimeRaw': '68.85',
        'N_CPUs': '2',
        'Mem_Req': '4G',
        'Mem_Used': '2.79G',
        'Partition': 'gpu_quad',
        'RequestedTime': '01:16:00'}
    """
    info_dict = {}
    info_dict['jobid'] = jobid
    
    def _batch_filt(line):
        return "batch" in line
    
    def _job_filt(line):
        exclude_terms = ["extern", "batch", "interactive", "RUNNING"]
        return all(term not in line for term in exclude_terms)

    # Use sacct to get information about the job
    cmd = f"O2sacct {jobid}"
    output = os.popen(cmd).read()
    lines = output.split('\n')

    # sacct info is split into two sections, batch and job, for unknown reasons
    job_line = lines[2]
    batch_line = lines[3]
    job_info = parse_sacct_line(job_line, "job")
    batch_info = parse_sacct_line(batch_line, "batch")
    
    # Combine the two dictionaries
    all_info = {**batch_info, **job_info}

    return all_info


def _validate_job_ids(jobids, how="squeue"):
    jobids_in = copy(jobids)
    if isinstance(jobids, list):
        jobids = [str(jobid) for jobid in jobids]
    elif isinstance(jobids, str) and "*" in jobids:
        cmd = f"{how} -u $USER | grep {jobids} | awk '{{print $1}}'"
        output = os.popen(cmd).read()
        jobids = output.split("\n")
        jobids = [jobid for jobid in jobids if jobid != ""]
        if len(jobids) == 0:
            raise ValueError(f"No jobs found matching {jobids_in}")
        if how == "sacct":
            jobids = [jobid.split(".")[0] for jobid in jobids]
            jobids = list(set(jobids))
    elif isinstance(jobids, str):
        jobids = [jobids]
    else:
        raise ValueError("jobids must be a list of jobids or a string with a grep pattern")
    return jobids


def get_job_info_df(jobids):
    """Get information about a list of jobs

    Parameters
    ----------
    jobids : list of str OR str with grep pattern
        List of jobids to get information about OR a grep pattern to match jobids to get information about

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing information about the jobs
    """
    import pandas as pd
    df = pd.DataFrame()
    jobids = _validate_job_ids(jobids, how="sacct")
    for jobid in jobids:
        job_info = get_job_info(jobid)
        df = pd.concat([df, pd.DataFrame(job_info, index=[0])], ignore_index=True)
    return df


def cancel_jobs(jobids, force=False):

    """Cancel a list of jobs, or all jobs matching a grep pattern

    Parameters
    ----------

    jobids : list of ints or strs, OR a str
        List of jobids to cancel OR a grep pattern to match jobids to cancel

    force : bool, optional
        If True, do not prompt for confirmation before cancelling jobs

    Returns
    -------
    jobids : list of str
        List of jobids that were cancelled
    """
    jobids = _validate_job_ids(jobids, how="squeue")
    
    if not force:
        print(f"About to cancel {len(jobids)} jobs. Continue? (y/n)")
        response = input()
        if response != "y":
            print("Exiting")
            return None

    for jobid in jobids:
        cmd = f"scancel {jobid}"
        os.system(cmd)

    return jobids


def format_time_from_sec(seconds):
    """Convert seconds to HH:MM:SS format

    Parameters
    ----------
    seconds : int
        Duration in seconds

    Returns
    -------
    str
        The duration represented in HH:MM:SS format
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def calculate_seconds_from_time_str(time_str):
    """Convert HH:MM:SS format to seconds

    Parameters
    ----------
    time_str : str
        Duration in HH:MM:SS format

    Returns
    -------
    int
        The duration in seconds
    """
    hours, minutes, seconds = map(int, time_str.split(":"))
    return hours*3600 + minutes*60 + seconds


def parse_squeue_output(output):
    """Parse the output of the squeue command into a dict with headers / values.

    """
    lines = output.strip().split('\n')
    headers = lines[0].split()
    job_data = {}
    
    for line in lines[1:]:
        fields = line.split(None, len(headers) - 1)
        job_info = dict(zip(headers, fields))
        job_id = job_info['JOBID']
        job_data[job_id] = job_info
    
    return job_data


def evaluate_job_status(jobids):
    """For a list of jobids, determine their status

    Parameters
    ----------
    jobids : list of str
        List of jobids to evaluate

    Returns
    -------
    job_info : dict
        Dictionary containing information about the jobs

    completed_jobs: list of str
        List of jobids that completed successfully
    
    failed_jobs : list of str
        List of jobids that failed

    timedout_jobs : list of str
        List of jobids that timed out
    """
    job_info = {}
    completed_jobs = []
    failed_jobs = []
    timedout_jobs = []
    for jobid in jobids:
        job_info[jobid] = {"sacct_info": get_job_info(jobid)}
        status = job_info[jobid]["sacct_info"]["Status"]
        if status == "FAILED":
            failed_jobs.append(jobid)
        elif status == "TIMEOUT" or status == "CANCELLED":
            timedout_jobs.append(jobid)
        elif status == "COMPLETED":
            completed_jobs.append(jobid)

    print(f"{len(completed_jobs)} completed: {completed_jobs}")
    print(f"{len(failed_jobs)} failed: {failed_jobs}")
    print(f"{len(timedout_jobs)} timed out: {timedout_jobs}")

    return job_info, completed_jobs, failed_jobs, timedout_jobs


def evaluate_resource_usage(jobids, plot=True):
    """Plot the time and memory usage for a list of jobs. Only considers COMPLETED jobs.

    Parameters
    ----------
    project_dir : str
        Path to the project directory

    jobids : list of str
        List of jobids to evaluate

    plot : bool
        If True, plot the results
    
    """

    if isinstance(jobids, str):
        jobids = [jobids]
    job_info = {}
    for jobid in jobids:
        job_info[jobid] = {"sacct_info": get_job_info(jobid)}

        if job_info[jobid]["sacct_info"]["Status"] != "COMPLETED":
            job_info.pop(jobid)
            continue

        # import pdb
        # pdb.set_trace()

        # Calculate fraction of time used
        elapsed_time = job_info[jobid]["sacct_info"]["ElapsedTime"]
        requested_time = job_info[jobid]["sacct_info"]["RequestedTime"]
        job_info[jobid]["time_fraction"] = calculate_seconds_from_time_str(elapsed_time) / calculate_seconds_from_time_str(requested_time)

        # Calculate fraction of memory used
        mem_req = job_info[jobid]["sacct_info"]["Mem_Req"]
        mem_used = job_info[jobid]["sacct_info"]["Mem_Used"]
        if mem_used == "nan" or mem_used == "0":
            mem_used = 0
        else:
            mem_used = float(mem_used[:-1])
        mem_req = float(mem_req[:-1])
        job_info[jobid]["mem_fraction"] = mem_used / mem_req

    if plot:
        # Generate a cute little summary figure
        # Figure with four subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs = axs.flatten()

        # sort the jobids by time usage
        sorted_jobids = sorted(job_info, key=lambda x: job_info[x]["time_fraction"])
        time_usages = [job_info[jobid]["time_fraction"] for jobid in sorted_jobids]
        mem_usages_sorted_wrt_time = [job_info[jobid]["mem_fraction"] for jobid in sorted_jobids]
        axs[0].bar(sorted_jobids, time_usages)
        axs[0].set_title("Sorted fractional time usage")
        axs[1].bar(sorted_jobids, mem_usages_sorted_wrt_time)
        axs[1].set_title("[memory]")
        
        # sort the jobids by memory usage, and plot
        sorted_jobids = sorted(job_info, key=lambda x: job_info[x]["mem_fraction"])
        mem_usages = [job_info[jobid]["mem_fraction"] for jobid in sorted_jobids]
        time_usages_sorted_wrt_mem = [job_info[jobid]["time_fraction"] for jobid in sorted_jobids]
        axs[2].bar(sorted_jobids, mem_usages)
        axs[2].set_title("Sorted fractional memory usage")
        axs[3].bar(sorted_jobids, time_usages_sorted_wrt_mem)
        axs[3].set_title("[time]")

        for ax in axs:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel("JobID")
            ax.set_ylabel("Fraction used")
            ax.set_ylim([0, 1])

        fig.tight_layout()

    return job_info
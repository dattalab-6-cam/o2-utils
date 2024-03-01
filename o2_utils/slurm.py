import os
from os.path import join, exists
import numpy as np
import sys
import re
import pdb

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
    if isinstance(jobids, list):
        jobids = [str(jobid) for jobid in jobids]
    elif isinstance(jobids, str) and "*" in jobids:
        cmd = f"{how} -u $USER | grep {jobids} | awk '{{print $1}}'"
        output = os.popen(cmd).read()
        jobids = output.split("\n")
        jobids = [jobid for jobid in jobids if jobid != ""]
        if len(jobids) == 0:
            raise ValueError(f"No jobs found matching {jobids}")
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
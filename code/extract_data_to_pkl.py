import tarfile
import os
import sys
import logging
import re
import threading
import time
import queue
import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
from shapely.geometry import LineString

def main(argv):
    #prefix_log = '/smartdata/px6680/Desktop/trumpf/'
    #prefix_data = '/gpfs/smartdata/ks6088/code/trumpf/Sampler/'
    # get the num of chunk
    num_job = argv[1]
    prefix_log = '../'
    prefix_data = '../data/'
    prefix_thread = 'thread_'
    num_core = 1
    # set logging
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(prefix_log+'log/extract_data.log')
    handler.setLevel(logging.INFO)
    formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formater)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.info('START THE PROGRAMM')
    # read filename
    filenames = os.listdir(prefix_data)
    filenames = [li for li in filenames if li.split('.')[-1] == 'tgz']
    filenames = [prefix_data+li for li in filenames]
    # distribute the file to job X
    # distribute file and create thread
    threads = []
    data = []
    num_chunk = int(len(filenames) / num_core)
    for index in range(num_core):
        if index == num_core - 1 :
            sub_filenames = filenames[index*num_chunk:]
        else:
            sub_filenames = filenames[index*num_chunk: (index+1)*num_chunk]
        threads.append(threading.Thread(target = extract_group, name = prefix_thread + str(index), args = (sub_filenames, prefix_thread+str(index), logger, )))
    # start and join the threads
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # save file
    logger.info('END THE PROGRAMM')
    
def extract_group(filenames, thread_name, logger = None):
    data = []
    for index, filename in enumerate(filenames):
        if (index+1) % 2 == 0:
            data = pd.concat(data, axis = 0)
            data.to_pickle(thread_name+'_'+str(index))
            data = []
        data.append(extract(filename, logger = logger))
    data = pd.concat(data, axis = 0)
    data.to_pickle(thread_name+'_end')
    # check the result before putting it into the queue
    #TODO

def extract(tar_path, target_path = 'tmp', mode = 'r:gz', logger = None):
    """
    input:
        tar_path string:
            path of tar file
        target_path string:
            path of target file
        mode string:
            mode of tar 
    """
    try:
        tar = tarfile.open(tar_path, mode)
        file_names = tar.getnames()
        # make sure that only two files contain
        #assert(len(file_names) == 2)
        # make sure the name of the files is right
        #TODO
        confs = []
        results = []
        if logger:
            logger.info('start extracting data from file: '+tar_path)
        file_names = [li for li in file_names if li.split('.')[-1] == 'txt']
        # logging start info
        for file_name in file_names:
            f = tar.extractfile(file_name)
            if 'configuration' in file_name:
                # use the configuration filename to get the result filename
                result_filename = file_name[:-18]+'.txt'
                # make sure the result file exist
                if result_filename not in file_names:
                    if logger:
                        logger.info('result file not found')
                    continue
                # extract the configuration file 
                f = tar.extractfile(file_name)
                conf = extract_configuration(file_name, f)
                # extract the result file
                f = tar.extractfile(result_filename)
                result = extract_result(file_name, f)
                # if mismatch 
                if len(conf) == len(result):
                    confs.append(conf)
                    results.append(result)
                elif len(conf) > len(result):
                    conf = conf[:len(result)]
                    confs.append(conf)
                    results.append(result)
                else:
                    if logger:
                        logger.error('Data mismatch in:'+ file_name)
                    else:
                        print('Data mismatch in:'+ file_name)
        tar.close()
        # combine conf and result
        conf = pd.concat(confs, axis = 0)
        result = pd.concat(results, axis = 0)
        target = pd.concat([conf, result], axis = 1)
        if logger:
            logger.info('end extracting')
        return target
    except:
        raise

def extract_configuration(filename, f):
    """
    input:
        filename string:
            the name of the file
        f file object:
            the file object
    """
    conf_columns = ['Jobid', 'Rot', 'List of Coordinates', 'Shapely Polygon']
    conf = pd.DataFrame(columns = conf_columns)
    for row_index, line in enumerate(f):
        l = line.decode("utf-8").strip()
        l = re.sub(r'\s+', ' ', l)
        ws = l.split(' ')
        assert(len(ws)>=2)
        conf.loc[row_index, 'Jobid'] = re.sub('[a-zA-Z/.]', '', filename)+str(row_index)
        conf.loc[row_index, 'Rot'] = ws[0]
        tmp_coor = fix_coordinate(ws[1:])
        conf.loc[row_index, 'List of Coordinates'] = tmp_coor
        conf.loc[row_index, 'Shapely Polygon'] = LineString(tmp_coor)
    return conf

def extract_result(filename, f):
    """
    input:
        filename string:
            the name of the file
        f file object:
            the file object    
    """
    result_columns = ['Metric1', 'Metric2', 'Metric3', 'Metric4']
    result = pd.DataFrame(columns = result_columns)
    for row_index, line in enumerate(f):
        l = line.decode("utf-8").strip()
        l = re.sub(r'\s+', ' ', l)
        ws = l.split(' ')
        if len(ws) != 6:
            print(ws)
        result.loc[row_index, 'Metric1'] = ws[2]
        result.loc[row_index, 'Metric2'] = ws[3]
        result.loc[row_index, 'Metric3'] = ws[4]
        result.loc[row_index, 'Metric4'] = ws[5]
    return result

def fix_coordinate(ws, logger = None):
    out = []
    out2 = []
    for w in ws:
        if w.count('.') == 1:
            out.append(float(w))
        elif w.count('.') > 1:
            # fix the problem
            tmp = []
            get_first(w, tmp)
            out.extend(tmp)
        else:
            # error??
            if logger:
                logger.error('coordinate error')
    if len(out)%2 != 0:
        logger.error('coordinate error')
    for i in range(0,len(out),2):
        out2.append((out[i],out[i+1]))
    return out2

def get_first(li, out):
    if li.count('.') > 1:
        add = li.index('.')+3
        out.append(float(li[:add]))
        get_first(li[add:], out)
    else:
        out.append(float(li))   

if __name__ == '__main__':
    main(sys.argv)

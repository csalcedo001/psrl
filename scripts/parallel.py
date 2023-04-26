import os
import subprocess
import yaml
import hashlib
import threading
import time
import datetime


def get_hash(obj):
    if type(obj) == dict:
        msg = str(sorted(obj.items()))
    else:
        msg = str(obj)
    
    obj_hash = hashlib.sha256(bytes(msg, 'utf-8')).hexdigest()[:8]

    return obj_hash

class ResourceAllocator():
    def __init__(self, config):
        self.lock = threading.Lock()

        self.resource_to_runs = {}
        self.id_to_pair = {}
        self.id_in_use = {}

        self._setup_config(config)


    
    def allocate(self):
        for id in self.id_to_pair:
            self.lock.acquire()

            if id in self.id_to_pair and not self.id_in_use.get(id, False):
                self.id_in_use[id] = True
                self.lock.release()

                return id
            else:
                self.lock.release()

        return None

    def free(self, id):
        self.lock.acquire()
        if id in self.id_to_pair:
            self.id_in_use[id] = False
        else:
            del self.id_in_use[id]
        self.lock.release()
    
    def get_resource(self, id):
        if self.id_in_use[id]:
            return self.id_to_pair[id][0]
        else:
            return None
    
    def _get_id_from_pair(self, id):
        return get_hash(get_hash(str(id[0])) + get_hash(str(id[1])))
    
    def _setup_config(self, config):
        # Validate config
        if type(config) == dict:
            for key, value in config.items():
                if type(key) != str:
                    raise ValueError("Keys in config represent a resource. Type must be str.")
                
                if type(value) != int:
                    raise ValueError("Values in config represent the number of runs. Type must be int.")
        elif type(config) == list:
            for resource in config:
                if type(resource) != int:
                    raise ValueError("Values in config represent the number of runs. Type must be int.")
        
            config_list = config
            config = {}
            for i, resource in enumerate(config_list):
                config[str(i)] = resource
        elif type(config) == int:
            config = {'0': config}


        # Set variables
        self.resource_to_runs = config
        
        id_to_pair = {}
        for resource in self.resource_to_runs:
            for run in range(self.resource_to_runs[resource]):
                pair = (resource, run)
                id = self._get_id_from_pair(pair)
                id_to_pair[id] = pair
                if id not in self.id_in_use:
                    self.id_in_use[id] = False
        
        self.id_to_pair = id_to_pair


class ParallelRunManager():
    def __init__(self, max_parallel_runs):
        self.allocator = ResourceAllocator(max_parallel_runs)
        self.id_to_thread = {}
    
    def queue(self, command_or_function, args):
        id = self.allocator.allocate()

        if id == None:
            return False

        if type(command_or_function) == str:
            command = command_or_function
            thread = threading.Thread(target=self._run_subprocess, args=(command, id))
            thread.start()
        elif callable(command_or_function):
            func = command_or_function
            thread = threading.Thread(target=self._run_function, args=(func, id, args))
            thread.start()

        self.id_to_thread[id] = thread

        return True

    def finish(self):
        for id in self.id_to_thread:
            self.id_to_thread[id].join()
    
    def _run_subprocess(self, command, id):
        resource = self.gpu_allocator.get_resource(id)

        environ = os.environ.copy()
        # environ["CUDA_VISIBLE_DEVICES"] = str(resource)
        
        subprocess.call(command, env=environ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.allocator.free(id)
    
    def _run_function(self, func, id, args):
        resource = self.allocator.get_resource(id)

        args['resource'] = resource
        func(args)

        self.allocator.free(id)
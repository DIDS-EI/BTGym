from btgym.utils.logger import log
import multiprocessing



def generate_one_way_communicator_pair():
    """创建一个单向的通信对,只能从发送方发送到接收方"""
    message_queue = multiprocessing.Queue()
    return OneWaySender(message_queue), OneWayReceiver(message_queue)


class OneWayCommunicator:
    def __init__(self, one_way_queue):
        self.one_way_queue = one_way_queue

    def set_owner(self, owner):
        self.owner = owner


class OneWaySender(OneWayCommunicator):
    def call(self, function_name, *args, **kwargs):
        self.one_way_queue.put((function_name, args, kwargs))


class OneWayReceiver(OneWayCommunicator): 
    def _deal_function(self):
        if self.one_way_queue.empty(): return

        function_name, args, kwargs = self.one_way_queue.get()
        # log(f'one way call received: {self.owner.__class__.__name__}.{function_name}')
        function = getattr(self.owner, function_name)
        function(*args, **kwargs)
        
    def deal_functions(self):
        while not self.one_way_queue.empty():
            self._deal_function()


def generate_two_way_communicator_pair():
    request_queue = multiprocessing.Queue()
    response_queue = multiprocessing.Queue()
    return RequestCommunicator(request_queue, response_queue), ResponseCommunicator(request_queue, response_queue)


class TwoWayCommunicator:
    def __init__(self,request_queue,response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue

    def set_owner(self, owner):
        self.owner = owner

class RequestCommunicator(TwoWayCommunicator):
    def call(self, function_name, *args, **kwargs): 
        self.request_queue.put(('-',function_name, args, kwargs))

    def request(self, function_name, *args, **kwargs): 
        self.request_queue.put(('=',function_name, args, kwargs))
        while True:
            if not self.response_queue.empty():
                return self.response_queue.get()
    
class ResponseCommunicator(TwoWayCommunicator):

    def deal_function(self):
        if self.request_queue.empty(): return

        type, function_name, args, kwargs = self.request_queue.get()
        log(f'two way call received: {self.owner.__class__.__name__}.{function_name}')
        function = getattr(self.owner, function_name)
        result = function(*args, **kwargs)

        if type == '=':
            self.response_queue.put(result)

        # log(f'two way call returned: {result}')



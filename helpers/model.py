from abc import ABCMeta, abstractmethod
import functools


def graph_node(function):
    
    @functools.wraps(function)
    def evaluate_once(self):
        function_name = function.__name__

        if function_name in self._nodes_added_to_graph:
            return self._nodes_added_to_graph[function_name]
        self._nodes_added_to_graph[function_name] = function(self)
        return self._nodes_added_to_graph[function_name]
        
    return evaluate_once


class TensorFlowBaseModel(metaclass=ABCMeta):
    
    def __init__(self):
        self._nodes_added_to_graph = {}
        self._add_nodes_to_graph()
    
    def _add_nodes_to_graph(self):
        for node in self._graph_nodes:
            node()
            
    @property
    @abstractmethod
    def _graph_nodes(self):
        return []

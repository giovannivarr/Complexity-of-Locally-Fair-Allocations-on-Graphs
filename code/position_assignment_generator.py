import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import is_isomorphic
from networkx.algorithms.traversal import dfs_tree
from typing import List, Dict, Tuple
from itertools import product
from mip import Model, xsum, BINARY, INTEGER, OptimizationStatus


def generate_assignments(agents: List[int], vertices: List[int]) -> Dict[int, int]:
    """
    Returns a dictionary containing a position assignment.

    :param agents: the list of agents.
    :param vertices: the list of vertices.
    :return: a dictionary containing a single position assignment.
    """
    assert len(agents) == len(vertices), 'The number of agents must be equal to the number of vertices.'

    if len(agents) == 0:
        yield {}
        return
    else:
        assignment = {}
        a = agents[0]
        for p in range(len(vertices)):
            assignment.update({a: vertices[p]})
            for rest in generate_assignments(agents[1:], vertices[:p] + vertices[p + 1:]):
                assignment.update(rest)
                yield assignment


def generate_all_assignments(agents: List[int], vertices: List[int]) -> List[Dict[int, int]]:
    """
    Returns a list containing all possible position assignments given a list of agents and one of vertices.

    :param agents: the list of agents.
    :param vertices: the list of vertices.
    :return: a list containing all possible position assignments, represented as a dictionary where each key is an agent
    and its value the position to which she is assigned.
    """
    assert len(agents) == len(vertices), 'The number of agents should be equal to the number of vertices.'

    assignments = []
    for x in generate_assignments(agents, vertices):
        y = {}
        y.update(x)
        assignments.append(y)
    return assignments


def compute_utilities(agents: List[int], utilities: Dict[int, np.ndarray],
                      item_allocation: Dict[int, List[int]]) -> Dict[int, List[float]]:
    """
    Computes the utilities of each agent with respect to each bundle which has been assigned in the item allocation.

    :param agents: the list of agents.
    :param utilities: the dictionary which represents the utility profile, each agent is a key and each value is an
    array of floats where the float in position i is the utility of item i for the key-agent.
    :param item_allocation: the dictionary which represents the fixed item allocation, where each key is an agents and
    its value the bundle assigned to such agent.
    :return: the dictionary where each key is an agents and its value a list of floats, where the i-th float is the
    utility of the bundle assigned to agent i for the key-agent.
    """
    utils = dict()
    for a in agents:
        u = []
        for allocation in item_allocation.values():
            u += [sum(utilities[a][i] for i in allocation)]
        utils.update({a: u})
    return utils


def find_envy_free_assignment(agents: List[int], assignments: List[Dict[int, int]],
                              utils: Dict[int, List[float]], graph: nx.Graph) \
        -> List[Dict[int, int]]:
    """
    Computes the list of envy-free position assignments given the set of agents, all the possible position assignments,
    the utilities of each agent for each assigned bundle and the social graph.

    :param agents: the list of agents.
    :param assignments: the list containing all position assignments, where each assignment is represented as a
    dictionary where a key is an agents and its value is the vertex to which such agents was assigned in that position
    assignment.
    :param utils: the dictionary which represents the utilities of each agent for each assigned bundle (recall that
    the item allocation is fixed in this case), a key is an agent and a value a list of floats where the float in
    position i is the utility of the bundle assigned to agent i for the key-agent.
    :param graph: the social graph.
    :return: a list containing all position assignments which are envy-free.
    """
    assert len(agents) == len(assignments[0].keys()), 'The assignments must assign the correct number of agents.'
    envy_bools = {agent: [True for i in agents] for agent in agents}
    for (agent, comparer) in product(agents, agents):
        envy_bools[agent][comparer] = True if utils[agent][agent] >= utils[agent][comparer] else False

    envy_free_allocations = []
    for assignment in assignments:
        flag = True
        for agent in agents:
            for (neighbor_agent, neighbor_vertex) in zip(assignment.keys(), assignment.values()):
                if agent == neighbor_agent or neighbor_vertex not in graph.neighbors(assignment[agent]):
                    continue
                if not (envy_bools[agent][neighbor_agent] and envy_bools[neighbor_agent][agent]):
                    flag = False
                    break
            if not flag:
                break

        if flag:
            envy_free_allocations += [assignment]

    return envy_free_allocations


def exists_envy_free_assignment(agents: List[int], assignments: List[Dict[int, int]],
                                utils: Dict[int, List[float]], graph: nx.Graph) -> bool:
    """
    Checks whether there is an envy-free position assignment given the set of agents, the utilities of each agent for
    each assigned bundle and the social graph.

    :param agents: the list of agents.
    :param assignments: the list containing all position assignments, where each assignment is represented as a
    dictionary where a key is an agents and its value is the vertex to which such agents was assigned in that position
    assignment.
    :param utils: the dictionary which represents the utilities of each agent for each assigned bundle (recall that
    the item allocation is fixed in this case), a key is an agent and a value a list of floats where the float in
    position i is the utility of the bundle assigned to agent i for the key-agent.
    :param graph: the social graph.
    :return: True if there is a local envy-free assignment and False otherwise.
    """
    assert len(agents) == len(assignments[0].keys()), 'The assignments must assign the correct number of agents.'

    envy_bools = {agent: [True for i in agents] for agent in agents}
    for (agent, comparer) in product(agents, agents):
        envy_bools[agent][comparer] = True if utils[agent][agent] >= utils[agent][comparer] else False

    for assignment in assignments:
        flag = True
        for agent in agents:
            for (neighbor_agent, neighbor_vertex) in zip(assignment.keys(), assignment.values()):
                if agent == neighbor_agent or neighbor_vertex not in graph.neighbors(assignment[agent]):
                    continue
                if not (envy_bools[agent][neighbor_agent] and envy_bools[neighbor_agent][agent]):
                    flag = False
                    break
            if not flag:
                break

        if flag:
            return True

    return False


def exists_envy_free_assignment_mip(agents: List[int], utils: Dict[int, List[float]], graph: nx.Graph) -> bool:
    """
    Checks whether there is an envy-free position assignment given the set of agents, the utilities of each agent for
    each assigned bundle and the social graph. Instead of iterating over all possible position assignments, this
    function uses an ILP.

    :param agents: the list of agents.
    :param utils: the dictionary which represents the utilities of each agent for each assigned bundle (recall that
    the item allocation is fixed in this case), a key is an agent and a value a list of floats where the float in
    position i is the utility of the bundle assigned to agent i for the key-agent.
    :param graph: the social graph.
    :return: True if there is a local envy-free assignment and False otherwise.
    """
    m, nodes = Model(), list(graph)

    for (agent, node) in product(agents, nodes):
        positions = [m.add_var(name='position_{}_{}'.format(agent, node), var_type=BINARY)]

    for agent in agents:
        m += xsum(m.var_by_name('position_{}_{}'.format(agent, node)) for node in nodes) == 1

    for node in nodes:
        m += xsum(m.var_by_name('position_{}_{}'.format(agent, node)) for agent in agents) == 1

    edges = [e for e in graph.edges]

    for (edge, agent1, agent2) in product(edges, agents, agents):
        node1, node2 = edge[0], edge[1]
        if utils[agent1][agent1] < utils[agent1][agent2]:
            m += m.var_by_name('position_' + str(agent1) + '_' + str(node1)) + \
                 m.var_by_name('position_' + str(agent2) + '_' + str(node2)) <= 1
            m += m.var_by_name('position_' + str(agent2) + '_' + str(node1)) + \
                 m.var_by_name('position_' + str(agent1) + '_' + str(node2)) <= 1

    return m.optimize() == OptimizationStatus.OPTIMAL


def find_proportional_assignment(agents: List[int], assignments: List[Dict[int, int]],
                                 utils: Dict[int, List[float]], graph: nx.Graph) -> List[Dict[int, int]]:
    """
    Computes the list of proportional position assignments given the set of agents, all the possible position
    assignments, the utilities of each agent for each assigned bundle and the social graph.

    :param agents: the list of agents.
    :param assignments: the list containing all position assignments, where each assignment is represented as a
    dictionary where a key is an agents and its value is the vertex to which such agents was assigned in that position
    assignment.
    :param utils: the dictionary which represents the utilities of each agent for each assigned bundle (recall that
    the item allocation is fixed in this case), a key is an agent and a value a list of floats where the float in
    position i is the utility of the bundle assigned to agent i for the key-agent.
    :param graph: the social graph.
    :return: a list containing all position assignments which are proportional.
    """
    assert len(agents) == len(assignments[0].keys()), 'The assignments must assign the correct number of agents'

    proportional_allocations = []
    for assignment in assignments:
        flag = True
        for agent in agents:
            mean = utils[agent][agent]
            for (neighbor_agent, neighbor_vertex) in zip(assignment.keys(), assignment.values()):
                if neighbor_vertex in graph.neighbors(assignment[agent]):
                    mean += utils[agent][neighbor_agent]

            mean /= len([m for m in graph.neighbors(assignment[agent])]) + 1
            if mean > utils[agent][agent]:
                flag = False
                break

        if flag:
            proportional_allocations += [assignment]

    return proportional_allocations


def exists_proportional_assignment(agents: List[int], assignments: List[Dict[int, int]],
                                utils: Dict[int, List[float]], graph: nx.Graph) -> bool:
    """
    Checks whether there is a proportional position assignment given the set of agents, the utilities of each agent for
    each assigned bundle and the social graph.

    :param agents: the list of agents.
    :param assignments: the list containing all position assignments, where each assignment is represented as a
    dictionary where a key is an agents and its value is the vertex to which such agents was assigned in that position
    assignment.
    :param utils: the dictionary which represents the utilities of each agent for each assigned bundle (recall that
    the item allocation is fixed in this case), a key is an agent and a value a list of floats where the float in
    position i is the utility of the bundle assigned to agent i for the key-agent.
    :param graph: the social graph.
    :return: True if there is a local proportional assignment and False otherwise.
    """
    assert len(agents) == len(assignments[0].keys()), 'The assignments must assign the correct ' \
                                                               'number of agents.'

    for assignment in assignments:
        flag = True
        for agent in agents:
            mean = utils[agent][agent]
            for (neighbor_agent, neighbor_vertex) in zip(assignment.keys(), assignment.values()):
                if neighbor_vertex in graph.neighbors(assignment[agent]):
                    mean += utils[agent][neighbor_agent]

            mean /= len([m for m in graph.neighbors(assignment[agent])]) + 1
            if mean > utils[agent][agent]:
                flag = False
                break

        if flag:
            return True

    return False


def find_envy_free_up_one_assignment(agents: List[int], assignments: List[Dict[int, int]],
                                  utils: Dict[int, List[float]], item_allocation: Dict[int, List[int]],
                                  item_utils: Dict[int, List[float]], graph: nx.Graph) -> List[Dict[int, int]]:
    """
    Computes the list of envy-free up to 1 item position assignments given the set of agents, all the possible position
    assignments, the utilities of each agent for each assigned bundle and the social graph.

    :param agents: the list of agents.
    :param assignments: the list containing all position assignments, where each assignment is represented as a
    dictionary where a key is an agents and its value is the vertex to which such agents was assigned in that position
    assignment.
    :param utils: the dictionary which represents the utilities of each assigned bundle for each agent, where each key
    is an agent and its value a list of floats where the float in position i is the utility of the bundle assigned
    to agent i for the key-agent.
    :param item_allocation: the fixed item allocation, represented as a dictionary where each key is an agent and the
    value is the agent's allocated bundle represented as a list.
    :param item_utils: the dictionary which represents the utilities of each item for each agent, where each key is
    an agent and its value a list of floats where the float in position i is the utility of item i for the key-agent.
    :param graph: the social graph.
    :return: a list containing all position assignments which are envy-free.
    """
    assert len(agents) == len(assignments[0].keys()), 'The assignments must assign the correct number of agents'

    up_one_bools = {agent: [1 for i in agents] for agent in agents}
    for (agent, comparer) in product(agents, agents):
        agent_bundle = {i: item_utils[agent][i] for i in item_allocation[agent]}
        comparer_bundle = {i: item_utils[agent][i] for i in item_allocation[comparer]}
        worst_chore, best_good = min(agent_bundle, key=agent_bundle.get), \
                                 max(comparer_bundle, key=comparer_bundle.get)
        up_one_diff = max(utils[agent][agent] - item_utils[agent][worst_chore] - utils[agent][comparer],
                          utils[agent][agent] + item_utils[agent][best_good] - utils[agent][comparer])
        up_one_bools[agent][comparer] = 1 if (up_one_diff >= 0 or utils[agent][agent] >= utils[agent][comparer]) else -1

    envy_free_up_one_allocations = []
    for assignment in assignments:
        flag = True
        for agent in agents:
            for (neighbor_agent, neighbor_vertex) in zip(assignment.keys(), assignment.values()):
                if agent == neighbor_agent or neighbor_vertex not in graph.neighbors(assignment[agent]):
                    continue
                if up_one_bools[agent][neighbor_agent] == -1 or up_one_bools[neighbor_agent][agent] == -1:
                    flag = False
                    break
            if not flag:
                break
        if flag:
            envy_free_up_one_allocations += [assignment]

    return envy_free_up_one_allocations


def exists_envy_free_up_one_assignment(agents: List[int], assignments: List[Dict[int, int]],
                                  utils: Dict[int, List[float]], item_allocation: Dict[int, List[int]],
                                  item_utils: Dict[int, List[float]], graph: nx.Graph) -> bool:
    """
    Computes the list of envy-free up to 1 item position assignments given the set of agents, all the possible position
    assignments, the utilities of each agent for each assigned bundle and the social graph.

    :param agents: the list of agents.
    :param assignments: the list containing all position assignments, where each assignment is represented as a
    dictionary where a key is an agents and its value is the vertex to which such agents was assigned in that position
    assignment.
    :param utils: the dictionary which represents the utilities of each assigned bundle for each agent, where each key
    is an agent and its value a list of floats where the float in position i is the utility of the bundle assigned
    to agent i for the key-agent.
    :param item_allocation: the fixed item allocation, represented as a dictionary where each key is an agent and the
    value is the agent's allocated bundle represented as a list.
    :param item_utils: the dictionary which represents the utilities of each item for each agent, where each key is
    an agent and its value a list of floats where the float in position i is the utility of item i for the key-agent.
    :param graph: the social graph.
    :return: True if there is a local envy-free up to one item assignment and False otherwise.
    """
    assert len(agents) == len(assignments[0].keys()), 'The assignments must assign the correct number of agents'

    up_one_bools = {agent: [1 for i in agents] for agent in agents}
    for (agent, comparer) in product(agents, agents):
        agent_bundle = {i: item_utils[agent][i] for i in item_allocation[agent]}
        comparer_bundle = {i: item_utils[agent][i] for i in item_allocation[comparer]}
        worst_chore, best_good = min(agent_bundle, key=agent_bundle.get), \
                                 max(comparer_bundle, key=comparer_bundle.get)
        up_one_diff = max(utils[agent][agent] - item_utils[agent][worst_chore] - utils[agent][comparer],
                          utils[agent][agent] + item_utils[agent][best_good] - utils[agent][comparer])
        up_one_bools[agent][comparer] = 1 if (up_one_diff >= 0 or utils[agent][agent] >= utils[agent][comparer]) else -1

    for assignment in assignments:
        flag = True
        for agent in agents:
            for (neighbor_agent, neighbor_vertex) in zip(assignment.keys(), assignment.values()):
                if agent == neighbor_agent or neighbor_vertex not in graph.neighbors(assignment[agent]):
                    continue
                if up_one_bools[agent][neighbor_agent] == -1 or up_one_bools[neighbor_agent][agent] == -1:
                    flag = False
                    break

            if not flag:
                break
        if flag:
            return True

    return False


def find_envy_agent_types(agents: List[int], utils: Dict[int, List[float]]) -> Dict[int, List[int]]:
    """
    Computes the agent-types and the agent-type of each of the given agents with respect to envy-freeness.

    :param agents: the list of agents.
    :param utils: the dictionary of utilities given a fixed item allocation, where each agent is a key and its value
    is a list of floats such that the i-th float is the utility of the bundle assigned to agent i for the key-agent.
    :return: a dictionary, where each key is an EF agent-type and its value is a list containing all agents of such
    type.
    """

    '''
    Computes the EF relation between each pair of agents and stores it in a matrix, where the value of cell (a1, a2)
    is 1 if a1 doesn't envy a2 and -1 otherwise. 
    '''
    envy_bools, types = {agent: [True for i in agents] for agent in agents}, dict()
    for (agent, comparer) in product(agents, agents):
        envy_bools[agent][comparer] = True if utils[agent][agent] >= utils[agent][comparer] else False

    for new_agent in agents:
        if len(types.keys()) == 0:
            types[len(types.keys())] = [new_agent]
            continue
        new_type_flag = True
        for t in types.keys():
            type_flag = True
            comparer = types[t][0]
            for agent in agents:
                if (agent != comparer and (
                    envy_bools[agent][new_agent] != envy_bools[agent][comparer] or
                    envy_bools[new_agent][agent] != envy_bools[comparer][agent]
                )) or (envy_bools[new_agent][comparer] != envy_bools[comparer][new_agent]):
                    type_flag = False
                    break
            if type_flag:
                types[t].append(new_agent)
                new_type_flag = False
                break
        if new_type_flag:
            types[len(types.keys())] = [new_agent]

    return types


def find_envy_up_one_agent_types(agents: List[int], utils: Dict[int, List[float]], item_utils: Dict[int, List[float]],
                                 item_allocation: Dict[int, List[int]]) -> \
                                    Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Computes the agent-types and the agent-type of each of the given agents with respect to envy-freeness up to one item
    and the difference in utilities of the representatives of each pair of agent-types.

    :param agents: the list of agents.
    :param utils: the dictionary of utilities given a fixed item allocation, where each agent is a key and its value
    is a list of floats such that the i-th float is the utility of the bundle assigned to agent i for the key-agent.
    :param item_utils: the dictionary which represents the utilities of each item for each agent, where each key is
    an agent and its value a list of floats where the float in position i is the utility of item i for the key-agent.
    :param item_allocation: the fixed item allocation, represented as a dictionary where each key is an agent and the
    value is the agent's allocated bundle represented as a list.
    :return: a tuple, where the first element is a dictionary where each key is an EF1 agent-type and its value a list
     of booleans such that the i-th boolean is True if and only if the representative of the key agent-type does not
     envy up one item the representative of the i-th agent-type, and the second element is a dictionary where each key
     is an agent-type and its value is a list containing all agents of such type.
    """
    up_one_bools, types = {agent: [1 for i in agents] for agent in agents}, dict()

    '''
    Computes the EF1 relation between each pair of agents and stores it in a matrix, where the value of cell (a1, a2)
    is 1 if a1 doesn't envy up to one item a2 and -1 otherwise. 
    '''
    for (agent, comparer) in product(agents, agents):
        agent_bundle = {i: item_utils[agent][i] for i in item_allocation[agent]}
        comparer_bundle = {i: item_utils[agent][i] for i in item_allocation[comparer]}
        worst_chore, best_good = min(agent_bundle, key=agent_bundle.get), \
                                 max(comparer_bundle, key=comparer_bundle.get)
        up_one_diff = max(utils[agent][agent] - item_utils[agent][worst_chore] - utils[agent][comparer],
                          utils[agent][agent] + item_utils[agent][best_good] - utils[agent][comparer])
        up_one_bools[agent][comparer] = 1 if (up_one_diff >= 0 or utils[agent][agent] >= utils[agent][comparer]) else -1

    for new_agent in agents:
        if len(types.keys()) == 0:
            types[len(types.keys())] = [new_agent]
            continue
        new_type_flag = True
        for t in types.keys():
            type_flag = True
            comparer = types[t][0]
            for agent in agents:
                if (comparer != agent and (
                    up_one_bools[new_agent][agent] != up_one_bools[comparer][agent] or
                    up_one_bools[agent][new_agent] != up_one_bools[agent][comparer]
                )) or (up_one_bools[new_agent][comparer] != up_one_bools[comparer][new_agent]):
                    type_flag = False
                    break
            if type_flag:
                types[t].append(new_agent)
                new_type_flag = False
                break
        if new_type_flag:
            types[len(types.keys())] = [new_agent]

    up_one_bools_types = {agent_type: [1 for i in types.keys()] for agent_type in types.keys()}
    for (agent_type, comparer_type) in product(types.keys(), types.keys()):
        up_one_bools_types[agent_type][comparer_type] = up_one_bools[types[agent_type][0]][types[comparer_type][0]]

    return up_one_bools_types, types


def find_vertex_types(T: nx.DiGraph) -> Dict[int, List[int]]:
    """
    Computes the vertex-types and the vertex-type of each vertex of a given tree. Input tree needs to be directed
    because otherwise the dfs will also go back to the root of the tree, instead of just going downwards.

    :param T: the input tree.
    :return: a dictionary, where each key is a vertex-type and its value is a list containing all nodes of that type.
    """
    assert nx.algorithms.tree.is_tree(T), 'Input graph should be a tree.'

    types = dict()

    for n in nx.nodes(T):
        if len(types.keys()) == 0:
            types[len(types.keys())] = [n]
            continue
        flag = True
        for t in types.keys():
            if is_isomorphic(dfs_tree(T, types[t][0]), dfs_tree(T, n)):
                types[t].append(n)
                flag = False
                break
        if flag:
            types[len(types.keys())] = [n]

    return types


def find_vertex_types_edges(T: nx.DiGraph, root: int, vertex_types: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    Finds for each ordered pair of vertex-types t, t' the number of times a vertex of type t is a parent of a vertex of
    type t'.

    :param T: the input tree.
    :param root: the integer which corresponds to the tree's root.
    :param vertex_types: the dictionary which represents the vertex-types, where each key is a vertex-type and its value
    is a list containing all vertices of such type.
    :return: a dictionary where each key is a vertex-type and its value a list of integers where the i-th integer is
    the number of times a vertex which types is the key-vertex-type is a parent of a vertex of type i.
    """
    vertex_types_edges = {i: [0 for j in vertex_types.keys()] for i in vertex_types.keys()}

    tree_heights = nx.shortest_path_length(T, root)

    for parent_type in vertex_types.keys():
        for children_type in vertex_types.keys():
            parent_vertex = vertex_types[parent_type][0]
            for children_vertex in vertex_types[children_type]:
                if tree_heights[parent_vertex] != tree_heights[children_vertex] - 1:
                    continue
                if (parent_vertex, children_vertex) in nx.edges(T):
                    vertex_types_edges[parent_type][children_type] += 1

    return vertex_types_edges


def parameterized_tree_lef_position_assignment(T: nx.DiGraph, root: int, agents: List[int],
                                           allocated_utilities: Dict[int, List[float]]) -> Model:
    """
    Solves an integer linear program (ILP) to determine whether there is an LEF position assignment in a tree.

    :param T: the input graph, which must be a tree.
    :param root: the integer which corresponds to the tree's root.
    :param agents: the list of agents.
    :param allocated_utilities: the dictionary which contains the utility for each agent, of each assigned bundle of a
    fixed item allocation. Each key corresponds to an agent and its value is a list of floats such that the i-th float
    is the utility of the bundle assigned to agent i for the key-agent.
    :return: an optimized mip.Model which constraints are the same as in the proof of Theorem 2.
    """
    assert type(T) == nx.classes.digraph.DiGraph, 'Input graph should be a directed graph.'
    assert nx.algorithms.tree.is_tree(T), 'Input graph should be a tree.'
    assert len([n for n in nx.nodes(T)]) == len(agents), \
        'Input tree should have a number of nodes equal to the number of input agents.'

    m = Model("Tree_ILP_Position_Assignment")

    agent_types = find_envy_agent_types(agents, allocated_utilities)

    vertex_types = find_vertex_types(T)

    vertex_types_edges = find_vertex_types_edges(T, root, vertex_types)

    edges = [m.add_var(name='edge_{}_{}_{}_{}'.format(a1, a2, v1, v2), var_type=INTEGER)
             for a1 in agent_types.keys() for a2 in agent_types.keys()
             for v1 in vertex_types.keys() for v2 in vertex_types.keys()]

    roots = [m.add_var(name='root_{}_{}'.format(a, v), var_type=BINARY)
             for a in agent_types.keys() for v in vertex_types.keys()]

    for v in vertex_types.keys():
        is_root_type = 1 if root in vertex_types[v] else 0
        m += xsum(m.var_by_name('root_' + str(a) + '_' + str(v)) for a in agent_types.keys()) == \
             is_root_type

    for a1 in agent_types.keys():
        m += \
            xsum(m.var_by_name('edge_' + str(a2) + '_' + str(a1) + '_' + str(v1) + '_' + str(v2))
                 for a2 in agent_types.keys() for v1 in vertex_types.keys() for v2 in vertex_types.keys()) + \
            xsum(m.var_by_name('root_' + str(a1) + '_' + str(v)) for v in vertex_types.keys()) == len(agent_types[a1])

        for v1 in vertex_types.keys():
            for v2 in vertex_types.keys():
                if root in vertex_types[v1]:
                    m += \
                        xsum(m.var_by_name('edge_' + str(a1) + '_' + str(a2) + '_' + str(v1) + '_' + str(v2))
                             for a2 in agent_types.keys()) - \
                        m.var_by_name('root_' + str(a1) + '_' + str(v1)) * vertex_types_edges[v1][v2] == 0
                else:
                    m += \
                        xsum(m.var_by_name('edge_' + str(a1) + '_' + str(a2) + '_' + str(v1) + '_' + str(v2))
                             for a2 in agent_types.keys()) - \
                        vertex_types_edges[v1][v2] * \
                        xsum(m.var_by_name('edge_' + str(a2) + '_' + str(a1) + '_' + str(v3) + '_' + str(v1))
                             for a2 in agent_types.keys() for v3 in vertex_types.keys()) == 0

                for a2 in agent_types.keys():
                    agent_a1, agent_a2 = agent_types[a1][0], agent_types[a2][0]
                    enviness_a1_a2 = allocated_utilities[agent_a1][agent_a1] - allocated_utilities[agent_a1][agent_a2]
                    enviness_a2_a1 = allocated_utilities[agent_a2][agent_a2] - allocated_utilities[agent_a2][agent_a1]

                    if enviness_a1_a2 < 0 or enviness_a2_a1 < 0:
                        m += m.var_by_name('edge_' + str(a1) + '_' + str(a2) + '_' + str(v1) + '_' + str(v2)) == 0

    return m


def parameterized_tree_lef1_position_assignment(T: nx.DiGraph, root: int, agents: List[int],
                                                utils: Dict[int, List[float]], item_utils: Dict[int, List[float]],
                                                item_allocation: Dict[int, List[int]]) -> Model:
    """
    Solves an integer linear program (ILP) to determine whether there is an LEF1 position assignment in a tree.

    :param T: the input graph, which must be a tree.
    :param root: the integer which corresponds to the tree's root.
    :param agents: the list of agents.
    :param utils: the dictionary of utilities given a fixed item allocation, where each agent is a key and its value
    is a list of floats such that the i-th float is the utility of the bundle assigned to agent i for the key-agent.
    :param item_utils: the dictionary which represents the utilities of each item for each agent, where each key is
    an agent and its value a list of floats where the float in position i is the utility of item i for the key-agent.
    :param item_allocation: the fixed item allocation, represented as a dictionary where each key is an agent and the
    value is the agent's allocated bundle represented as a list.
    :return: an optimized mip.Model which constraints are the same as in the proof of Theorem 8.
    """
    assert type(T) == nx.classes.digraph.DiGraph, 'Input graph should be a directed graph.'
    assert nx.algorithms.tree.is_tree(T), 'Input graph should be a tree.'
    assert len([n for n in nx.nodes(T)]) == len(agents), \
        'Input tree should have a number of nodes equal to the number of input agents.'

    m = Model("Tree_ILP_Position_Assignment")

    up_one_output = find_envy_up_one_agent_types(agents, utils, item_utils, item_allocation)
    up_one_bools, agent_types = up_one_output[0], up_one_output[1]

    vertex_types = find_vertex_types(T)

    vertex_types_edges = find_vertex_types_edges(T, root, vertex_types)

    edges = [m.add_var(name='edge_{}_{}_{}_{}'.format(a1, a2, v1, v2), var_type=INTEGER)
             for a1 in agent_types.keys() for a2 in agent_types.keys()
             for v1 in vertex_types.keys() for v2 in vertex_types.keys()]

    roots = [m.add_var(name='root_{}_{}'.format(a, v), var_type=BINARY)
             for a in agent_types.keys() for v in vertex_types.keys()]

    for v in vertex_types.keys():
        is_root_type = 1 if root in vertex_types[v] else 0
        m += xsum(m.var_by_name('root_' + str(a) + '_' + str(v)) for a in agent_types.keys()) == \
             is_root_type

    for a1 in agent_types.keys():
        m += \
            xsum(m.var_by_name('edge_' + str(a2) + '_' + str(a1) + '_' + str(v1) + '_' + str(v2))
                 for a2 in agent_types.keys() for v1 in vertex_types.keys() for v2 in vertex_types.keys()) + \
            xsum(m.var_by_name('root_' + str(a1) + '_' + str(v)) for v in vertex_types.keys()) == len(agent_types[a1])

        for v1 in vertex_types.keys():
            for v2 in vertex_types.keys():
                if root in vertex_types[v1]:
                    m += \
                        xsum(m.var_by_name('edge_' + str(a1) + '_' + str(a2) + '_' + str(v1) + '_' + str(v2))
                             for a2 in agent_types.keys()) - \
                        m.var_by_name('root_' + str(a1) + '_' + str(v1)) * vertex_types_edges[v1][v2] == 0
                else:
                    m += \
                        xsum(m.var_by_name('edge_' + str(a1) + '_' + str(a2) + '_' + str(v1) + '_' + str(v2))
                             for a2 in agent_types.keys()) - \
                        vertex_types_edges[v1][v2] * \
                        xsum(m.var_by_name('edge_' + str(a2) + '_' + str(a1) + '_' + str(v3) + '_' + str(v1))
                             for a2 in agent_types.keys() for v3 in vertex_types.keys()) == 0

                for a2 in agent_types.keys():
                    agent_a1, agent_a2 = agent_types[a1][0], agent_types[a2][0]

                    if up_one_bools[a1][a2] == -1 or up_one_bools[a2][a1] < 0:
                        m += m.var_by_name('edge_' + str(a1) + '_' + str(a2) + '_' + str(v1) + '_' + str(v2)) == 0

    return m

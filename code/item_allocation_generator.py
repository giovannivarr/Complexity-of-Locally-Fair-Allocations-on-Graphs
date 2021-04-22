import random
from mip import Model, xsum, minimize, maximize, BINARY, CONTINUOUS
from typing import List, Dict
from itertools import product


def max_utilitarian_welfare_allocation(utilities: Dict[int, List[float]], m: Model) -> None:
    """
    Computes (one of) the item allocation(s) which maximizes utilitarian welfare, returning the optimized model.

    :param utilities: the dictionary representing the utility profile, where each key is an agent and its value an array
    of floats such that the i-th float is the utility of the i-th item for the key-agent.
    :param m: the MIP model which represents the integer linear program.
    """
    agents, items = len(utilities), len(list(utilities.values())[0])

    m.objective = maximize(xsum(utilities[agent][item] * m.var_by_name('assign_{}_{}'.format(item, agent))
                                for item in range(items) for agent in range(agents)))

    m.optimize()


def min_enviness_allocation(utilities: Dict[int, List[float]], m: Model) -> None:
    """
    Computes (one of) the item allocation(s) which minimizes global enviness (observe we only sum enviness when it is
    larger than 0).

    :param utilities: the dictionary representing the utility profile, where each key is an agent and its value an array
    of floats such that the i-th float is the utility of the i-th item for the key-agent.
    :param m: MIP model to optimize.
    :return: a dictionary mapping to each agent the bundle which has been assigned to her so that enviness is minimized.
    """
    agents, items = len(utilities), len(list(utilities.values())[0])

    dummies = [m.add_var(name='dummy_{}_{}'.format(agent1, agent2), var_type=CONTINUOUS)
               for agent2 in range(agents) for agent1 in range(agents)]

    m.objective = minimize(xsum(m.var_by_name('dummy_{}_{}'.format(agent1, agent2))
                                for agent2 in range(agents) for agent1 in range(agents)))

    for (agent1, agent2) in product(range(agents), range(agents)):
        m += m.var_by_name('dummy_{}_{}'.format(agent1, agent2)) >= 0

        m += m.var_by_name('dummy_{}_{}'.format(agent1, agent2)) >= \
             sum(utilities[agent1][item] * m.var_by_name('assign_{}_{}'.format(item, agent2)) for item in range(items))\
            - sum(utilities[agent1][item] * m.var_by_name('assign_{}_{}'.format(item, agent1)) for item in range(items))

    m.optimize()


def min_unproportionality_allocation(utilities: Dict[int, List[float]], m: Model) -> None:
    """
    Computes (one of) the item allocation(s) which minimizes global unproportionality (observe we only sum
    unproportionality when it is larger than 0).

    :param utilities: the dictionary representing the utility profile, where each key is an agent and its value an array
    of floats such that the i-th float is the utility of the i-th item for the key-agent.
    :param m: the MIP model to optimize.
    :return: a dictionary mapping to each agent the bundle which has been assigned to her so that unproportionality
     is minimized.
    """
    agents, items = len(utilities), len(list(utilities.values())[0])

    dummies = [m.add_var(name='dummy_{}'.format(agent), var_type=CONTINUOUS) for agent in range(agents)]

    m.objective = minimize(xsum(m.var_by_name('dummy_{}'.format(agent)) for agent in range(agents)))

    for agent in range(agents):
        m += m.var_by_name('dummy_{}'.format(agent)) >= 0

        m += m.var_by_name('dummy_{}'.format(agent)) >= (sum(utilities[agent][item] for item in range(items)) / agents)\
        - (sum(utilities[agent][item] * m.var_by_name('assign_{}_{}'.format(item ,agent)) for item in range(items)))

    m.optimize()


def random_allocation(agents: List[int], items: List[int]) -> Dict[int, List[int]]:
    """
    Generates a random allocation such that each item is assigned to one agent and each agent is assigned at least one
    item.

    :param agents: the integer representing the number of agents.
    :param items: the integer representing the number of items.
    :return: a dictionary mapping to each agent the bundle which has been randomly assigned to her.
    """
    temp_agents, allocation = agents[:], {agent: [] for agent in agents}

    while temp_agents:  # ensures that each agent is assigned at least one item
        random_agent, random_item = random.choice(temp_agents), random.choice(items)
        allocation[random_agent] += [random_item]
        temp_agents.remove(random_agent)
        items.remove(random_item)

    while items:
        random_agent, random_item = random.choice(agents), random.choice(items)
        allocation[random_agent] += [random_item]
        items.remove(random_item)

    return allocation


def generate_allocation(utilities: Dict[int, List[float]], criterion: str) -> Dict[int, List[int]]:
    """
    :param utilities: the dictionary representing the utility profile, where each key is an agent and its value an array
    of floats such that the i-th float is the utility of the i-th item for the key-agent.
    :param criterion: the string which represents the criteria by which the item allocation is performed. This can be
    either 'max_utilitarian', 'min_enviness', 'min_unproportional' and 'random'.
    :return: a dictionary where each key is an agent and its corresponding value is the list of item which were assigned
    to her.
    """
    assert len(set(len(utilities[a]) for a in utilities.keys())) == 1, 'All agents should have a utility for the same '\
                                                                       'number of items.'

    assert len(utilities) <= len(list(utilities.values())[0]), 'The number of agents should be smaller than or equal ' \
                                                               'to the number of items.'

    criteria = {'max_utilitarian': max_utilitarian_welfare_allocation, 'min_enviness': min_enviness_allocation,
                'min_unproportionality': min_unproportionality_allocation, 'random': random_allocation}

    if criterion == 'random':
        allocation = criteria[criterion](list(utilities.keys()), list(range(len(list(utilities.values())[0]))))

    elif criterion in criteria:
        m = Model(name=criterion)
        agents, items = len(utilities), len(list(utilities.values())[0])

        allocations = [m.add_var(name='assign_{}_{}'.format(item, agent), var_type=BINARY)
                       for item in range(items) for agent in range(agents)]

        for item in range(items):
            m += xsum(m.var_by_name('assign_{}_{}'.format(item, agent)) for agent in range(agents)) == 1

        for agent in range(agents):
            m += xsum(m.var_by_name('assign_{}_{}'.format(item, agent)) for item in range(items)) >= 1

        criteria[criterion](utilities, m)

        allocation = {}
        for agent in range(agents):
            items_assigned = []
            for item in range(items):
                if m.var_by_name('assign_{}_{}'.format(item, agent)).x == 1:
                    items_assigned += [item]
            allocation.update({agent: items_assigned})

    else:
        raise Exception("Criterion {} has not been implemented yet.".format(criterion))

    return allocation

from enum import Enum
import pandas as pd
import numpy as np


class EventCombination(Enum):
    ALL_EVENTS = 1
    ANY_EVENT = 2
    NO_EVENTS = 3


def estimate_p_x_y(random_variates: pd.DataFrame, x: str, y: str):
    """Estimates P(X|Y) from counts."""
    count_y = random_variates[y].sum()
    count_x_and_y = (random_variates[x] * random_variates[y]).sum()

    return (count_x_and_y, count_y, count_x_and_y / count_y)


def generate_new_event(
    random_variates: pd.DataFrame, ec: EventCombination, variables_events: [tuple]
):
    """Generate new event array based on specified list of (variable,event) pairs and a combination (all, any, no)."""
    num_events = len(variables_events)
    event_array = np.column_stack(
        [np.array((random_variates[v] == e).astype(int)) for v, e in variables_events]
    )
    event_array_sum = event_array.sum(axis=1)

    if ec == EventCombination.ALL_EVENTS:
        return (event_array_sum == num_events).astype(int)
    elif ec == EventCombination.ANY_EVENT:
        return (event_array_sum > 0).astype(int)
    elif ec == EventCombination.NO_EVENTS:
        return (event_array_sum == 0).astype(int)
    else:
        raise AssertionError(
            f"Invalid input: {ec} - expected valid EventCombination input"
        )


def estimate_p_x_y_generalised(
    random_variates: pd.DataFrame,
    ec_x: EventCombination,
    variables_events_x: [tuple],
    ec_y: EventCombination,
    variables_events_y: [tuple],
):
    """Estimates P(X|Y) for generalised events X,Y."""
    event_x = generate_new_event(random_variates, ec_x, variables_events_x)
    event_y = generate_new_event(random_variates, ec_y, variables_events_y)
    event_x_and_y = event_x * event_y

    count_y = event_y.sum()
    count_x_and_y = event_x_and_y.sum()

    return (count_x_and_y, count_y, count_x_and_y / count_y)

# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
import copy


def retrieve_dispatch_times_given_list_of_dispatches(
        number_of_vehicles: int,
        makespan_to_consider: float,
        g_dispatch_times_function: callable,
        order_arrival_times: list[float],
        batches_dispatched_by_each_vehicle: list[tuple[int, int]]
        ):
    """
    This function receives as input the detailed batches that each vehicle
    dispatches to, based on the optimal solution that is below the 
    'makespan_to_consider'. 
    Based on that, it tries to schedule the batches as late as it can, and 
    returns the schedule of those batches for each vehicle.
    In case two batches of the same vehicle can be combined, it does so. In
    general, due to economies of scale of function g, it will not occur, as long
    as the solution is optimal with respect to distance driven; but this function
    is built to support non-optimal solutions.
    """
    start_and_end_for_dispatches = []
    cardinality_of_dispatches = []
    # Iterate over each vehicle
    for i in range(0, number_of_vehicles):
        
        # We deepcopy the batches done by this vehicle
        dispatches_list = copy.deepcopy(batches_dispatched_by_each_vehicle[i])
        makespan_to_measure_against = makespan_to_consider
        
        # We add the lists with data for this vehicle
        start_and_end_for_dispatches.append([])
        cardinality_of_dispatches.append([])
        
        # We iterate for all dispatches, until we have processed them all
        while len(dispatches_list) >= 1:
            # the -1 at the end accounts for the fact indices have a +1
            minimum_start_time_for_dispatch = order_arrival_times[
                dispatches_list[-1][1] - 1]
            
            # start time for the batch, based on the dispatch time given by cardinality
            provisional_start_time = makespan_to_measure_against \
                - g_dispatch_times_function(
                    # cardinality
                    dispatches_list[-1][1] - dispatches_list[-1][0] + 1)
            
            # Note that because of feasibility (as batches_dispatched_by_each_vehicle
            # is feasible under makespan_to_consider), the first statement of the
            # while is always true, when we initially enter the while.
            # The second might not be.
            while provisional_start_time >= minimum_start_time_for_dispatch \
                        and len(dispatches_list) >= 2:
                
                provisional_start_time_if_combining_batches = \
                    makespan_to_measure_against - g_dispatch_times_function(
                        # cardinality when combining batches
                        dispatches_list[-1][1] - dispatches_list[-2][0] + 1
                    )

                # If this happens, it means we can batch 2 dispatches in one
                # because this vehicle is not tight in terms of makespan
                if provisional_start_time_if_combining_batches >= minimum_start_time_for_dispatch:
                    # We update the last dispatch to include all orders from dispatch -1
                    dispatches_list[-1][0] = dispatches_list[-2][0]
                    # we pop dispatch -2 since we combined them
                    dispatches_list.pop(-2)

                # We cannot batch the 2 dispatches, so we break the while and 
                # go to the next step
                else:
                    break
        
            # We recompute the start time; with the definitive batch
            provisional_start_time = makespan_to_measure_against - \
                g_dispatch_times_function(
                    dispatches_list[-1][1] - dispatches_list[-1][0] + 1)
            
            # Update our outputs; the index [-1] indicates the last vehicle
            start_and_end_for_dispatches[-1].insert(0, [provisional_start_time, makespan_to_measure_against] )
            cardinality_of_dispatches[-1].insert(0, dispatches_list[-1][1] - dispatches_list[-1][0] + 1)
            
            # The next dispatch will be measured against the start of the 
            # dispatch that goes after 
            makespan_to_measure_against=provisional_start_time

            # We remove the last dispatch from the queue to be processed
            dispatches_list.pop(-1)

    # After processing all dispatches and all vehicles we return the results
    return start_and_end_for_dispatches, cardinality_of_dispatches


def plotting_dispatches_results_for_three_different_fleet_sizes(
        order_arrival_times: list[int],
        # We use fleet size = 1 for our figures; this is why annotation is just 2 nested lists
        # instead of 3 as the two other fleet sizes below
        schedule_dispatches_fleet_size_1: list[list[float]],
        cardinality_of_dispatches_fleet_size_1: list[list[float]],
        number_of_orders_done_with_fleet_size_1: int,
        fleet_size_1: int,
        schedule_dispatches_fleet_size_2: list[list[list[float]]],
        cardinality_of_dispatches_fleet_size_2: list[list[list[float]]],
        number_of_orders_done_with_fleet_size_2: int,
        fleet_size_2: int,
        schedule_dispatches_fleet_size_3: list[list[list[float]]], 
        cardinality_of_dispatches_fleet_size_3: list[list[list[float]]],
        number_of_orders_done_with_fleet_size_3: int,
        fleet_size_3: int,
        makespan_for_figure: float,
        # start of day is 9 am
        start_of_day: float=9,
        unit_arrivals_to_hours: float=0.1,
        name: str="dispatches_for_three_different_fleet_sizes"
        ):
    """
    This function plots in a single graph the structures of the dispatches for
    three different fleet sizes. This corresponds to Figure 2 in the paper.

    An assumption to use this function is that the fleet sizes need to be 
    increasing; and the code also leverages that the first fleet size is 1.
    """
    # Creating the figure
    fig, ax = plt.subplots()

    # Add the arrival ticks in x axis; for all order arrivals
    # Arrivals that can be done by the first fleet size
    x_vector = [start_of_day]
    y_vector = [0] 
    for i in range(number_of_orders_done_with_fleet_size_1):
        x_vector.append(order_arrival_times[i] * unit_arrivals_to_hours + start_of_day)
        y_vector.append(0)
    ax.plot(x_vector, y_vector, '|', color='black', linestyle="None")

    
    # Arrivals that can be done by the second fleet size but not the first
    x_vector = []
    y_vector = []
    for i in range(number_of_orders_done_with_fleet_size_1, number_of_orders_done_with_fleet_size_2): ###64+1 first sddq1_1, 62+1 q1_2, 67+1 sddq1_3
        x_vector.append(order_arrival_times[i] * unit_arrivals_to_hours + start_of_day)
        y_vector.append(0)
    ax.plot(x_vector, y_vector, '|', color='red', linestyle="None")

    
    # Arrivals that can be done by the third fleet size but not the second
    x_vector = []
    y_vector = []
    for i in range(number_of_orders_done_with_fleet_size_2, number_of_orders_done_with_fleet_size_3):  ### 70+1 sddq1_1, 69+1 q1_2, 75+1 sddq1_3
        x_vector.append(order_arrival_times[i] * unit_arrivals_to_hours + start_of_day)
        y_vector.append(0)
    ax.plot(x_vector, y_vector, '|', color='blue', linestyle="None")


    # Adjust the values to be in terms of hours; each unit of time is in hours
    for values in [schedule_dispatches_fleet_size_1, 
                   schedule_dispatches_fleet_size_2, 
                   schedule_dispatches_fleet_size_3]:
        for number_vehicle in range(0, len(values)):
            for dispatch in range(0, len(values[number_vehicle])):
                if type(values[number_vehicle][dispatch]) == list:
                    for entry in range(0, len(values[number_vehicle][dispatch])):
                        values[number_vehicle][dispatch][entry] = \
                            values[number_vehicle][dispatch][entry] \
                            * unit_arrivals_to_hours + start_of_day
                # In this scenario there is just 1 vehicle, so we adjust accordingly
                else:
                    values[number_vehicle][dispatch] = values[number_vehicle][dispatch] \
                        * unit_arrivals_to_hours + start_of_day

    # Creating the dispatch lines for the first fleet size = 1 vehicle
    for i in range(0, len(schedule_dispatches_fleet_size_1)):
        verts = []
        code = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        # start of edge
        verts.append((schedule_dispatches_fleet_size_1[i][0], 0))
        # midpoint of edge
        verts.append((schedule_dispatches_fleet_size_1[i][0]/2 + schedule_dispatches_fleet_size_1[i][1]/2, 0.15))
        # end of edge
        verts.append((schedule_dispatches_fleet_size_1[i][1], 0))
        path = Path(verts, code)
        patch = mpatches.FancyArrowPatch(path=path,
                                arrowstyle="-|>,head_length=3,head_width=3")
        ax.add_patch(patch)
        # add the cardinality of dispatch
        ax.text(schedule_dispatches_fleet_size_1[i][0]/2 + schedule_dispatches_fleet_size_1[i][1]/2, 
                0.09, str(cardinality_of_dispatches_fleet_size_1[i]))

    # Creating the dispatch lines for the second fleet size
    verts = []
    for vehicle in range(0, len(schedule_dispatches_fleet_size_2)):
        for i in range(0, len(schedule_dispatches_fleet_size_2[vehicle])):
            verts = []
            code = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            # start of edge
            verts.append((schedule_dispatches_fleet_size_2[vehicle][i][0], 0.5 + vehicle/3))
            # midpoint of edge
            verts.append((schedule_dispatches_fleet_size_2[vehicle][i][0]/2 \
                + schedule_dispatches_fleet_size_2[vehicle][i][1]/2, 
                # y-coordinate
                0.5 + vehicle/3 + 0.15))
            # end of edge
            verts.append((schedule_dispatches_fleet_size_2[vehicle][i][1], 0.5 + vehicle/3))
            path = Path(verts, code)
            patch = mpatches.FancyArrowPatch(path=path,
                                arrowstyle="-|>,head_length=3,head_width=3",color='red')
            ax.add_patch(patch)
            # add the cardinality of dispatch
            ax.text(schedule_dispatches_fleet_size_2[vehicle][i][0]/2\
                + schedule_dispatches_fleet_size_2[vehicle][i][1]/2,
                0.5 + vehicle/3 + 0.09, 
                str(cardinality_of_dispatches_fleet_size_2[vehicle][i]))
    
    # Creating the dispatch lines for the third fleet size
    verts = []
    for vehicle in range(0, len(schedule_dispatches_fleet_size_3)):
        for i in range(0, len(schedule_dispatches_fleet_size_3[vehicle])):
            verts = []
            code = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            # start of edge
            verts.append((schedule_dispatches_fleet_size_3[vehicle][i][0], 1.5 + vehicle/3))
            # midpoint of edge
            verts.append((schedule_dispatches_fleet_size_3[vehicle][i][0]/2 \
                + schedule_dispatches_fleet_size_3[vehicle][i][1]/2, 
                1.5 + vehicle/3 + 0.15))
            # end of edge
            verts.append((schedule_dispatches_fleet_size_3[vehicle][i][1], 1.5 + vehicle/3))
            path = Path(verts, code)
            patch = mpatches.FancyArrowPatch(path=path,
                                arrowstyle="-|>,head_length=3,head_width=3",color='blue')
            ax.add_patch(patch)
            # add the cardinality of dispatch
            ax.text(schedule_dispatches_fleet_size_3[vehicle][i][0]/2 \
                + schedule_dispatches_fleet_size_3[vehicle][i][1]/2,
                1.5 + vehicle/3 + 0.09, 
                str(cardinality_of_dispatches_fleet_size_3[vehicle][i]))
    
    # Legend
    ax.plot(0,0,",-",color="black",label=f"Solution with {fleet_size_1} vehicle, {number_of_orders_done_with_fleet_size_1} orders")
    ax.plot(0,0,",-",color="red",label=f"Solution with {fleet_size_2} vehicles, {number_of_orders_done_with_fleet_size_2} orders")
    ax.plot(0,0,",-",color="blue",label=f"Solution with {fleet_size_3} vehicles, {number_of_orders_done_with_fleet_size_3} orders")
    
    # epsilon makes the graph to extend a bit so it is easier to see x avis
    epsilon = 0.1

    # Add scaling for x and y axis
    ax.set_xlim(start_of_day, 
                makespan_for_figure * unit_arrivals_to_hours + start_of_day + epsilon)
    ax.set_ylim(0, 4)
    ax.yaxis.set_ticklabels([])

    # Labels for the axis
    ax.set_xlabel("Hour",color="black",fontsize=11)
    ax.set_ylabel("Vehicles' Dispatches and Cardinalities", color="black", fontsize=11)

    # Create legend
    plt.legend(loc=2)

    # Show and save figure
    plt.show()
    fig.savefig(name+".pdf",
                format='pdf',
                dpi=300,
                bbox_inches='tight')
    return verts
import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import itertools
import math
import datetime


@st.cache_data
def load_data():
    return pd.read_csv("data/synthetic_user_data.csv")


@st.cache_data
def load_ride_data():
    return pd.read_csv("data/ride_capacity.csv")


@st.cache_data
def load_waiting_data():
    return pd.read_csv("data/waiting_times.csv")


@st.cache_data
def load_pathing_schedules():
    return pd.read_csv("data/pathing_scheduled.csv")


@st.cache_data
def load_attractions_location():
    return pd.read_csv(
        "data/link_attraction_park.csv", sep=";"
    )  # which rides are in which park


synthetic_users = load_data()
ride_capacity = load_ride_data()
waiting_df = load_waiting_data()
schedule_df = load_pathing_schedules()
link_attractions_df = load_attractions_location()


class ThemeParkGraph:
    def __init__(self, attractions):
        self.G = nx.Graph()
        self.attractions = attractions
        self.G.add_nodes_from(attractions)
        self.manual_distances = {}

    def add_manual_distance(self, attraction1, attraction2, distance):
        key = tuple(sorted([attraction1, attraction2]))
        self.manual_distances[key] = distance

    def get_distance(self, attr1, attr2):
        key = tuple(sorted([attr1, attr2]))
        return self.manual_distances.get(key, random.randint(50, 300))

    def generate_paths(self, connections_per_attraction=(2, 4)):
        min_connections, max_connections = connections_per_attraction
        paths = []

        for i, attr1 in enumerate(self.attractions):
            num_connections = random.randint(min_connections, max_connections)
            possible_connections = self.attractions[i + 1 :]

            if len(possible_connections) > 0:
                num_connections = min(num_connections, len(possible_connections))
                for attr2 in random.sample(possible_connections, num_connections):
                    distance = self.get_distance(attr1, attr2)
                    paths.append((attr1, attr2, distance))

        self.G.add_weighted_edges_from(paths)

    def stress_majorization_layout(self):
        """Create a layout that tries to respect actual distances"""

        def stress(pos_flat):
            pos = pos_flat.reshape(-1, 2)
            stress_sum = 0

            for u, v, d in self.G.edges(data=True):
                u_idx = list(self.G.nodes()).index(u)
                v_idx = list(self.G.nodes()).index(v)
                target_dist = d["weight"]
                actual_dist = np.linalg.norm(pos[u_idx] - pos[v_idx])
                stress_sum += (target_dist - actual_dist) ** 2

            return stress_sum

        # Start with random positions
        initial_pos = np.random.rand(len(self.G.nodes()), 2)

        # Optimize positions
        result = minimize(stress, initial_pos.flatten(), method="L-BFGS-B")
        final_pos = result.x.reshape(-1, 2)

        # Convert to dictionary format
        return {node: pos for node, pos in zip(self.G.nodes(), final_pos)}

    def visualize(self, use_distance_layout=True, figsize=(20, 20)):
        fig, ax = plt.subplots(figsize=figsize)  # Create a figure and axis

        # Choose layout method
        if use_distance_layout:
            pos = self.stress_majorization_layout()
        else:
            pos = nx.kamada_kawai_layout(self.G)

        # Scale positions to be more readable
        pos_scaled = {node: (coords * 2 - 1) for node, coords in pos.items()}

        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, pos_scaled, node_color="lightblue", node_size=2000, alpha=0.7, ax=ax
        )

        # Draw edges with colors based on distance
        edges = self.G.edges(data=True)
        weights = [d["weight"] for (_, _, d) in edges]
        min_weight, max_weight = min(weights), max(weights)

        # Normalize weights for coloring
        norm = plt.Normalize(vmin=min_weight, vmax=max_weight)
        cmap = plt.cm.viridis
        edge_colors = [cmap(norm(w)) for w in weights]

        nx.draw_networkx_edges(
            self.G, pos_scaled, width=2, edge_color=edge_colors, alpha=0.6, ax=ax
        )

        # Add node labels
        labels = {node: "\n".join(node.split()) for node in self.G.nodes()}
        nx.draw_networkx_labels(
            self.G, pos_scaled, labels, font_size=8, font_weight="bold", ax=ax
        )

        # Add edge labels
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(
            self.G, pos_scaled, edge_labels, font_size=6, ax=ax
        )

        ax.set_title(
            "Euro Park Attractions Graph\nShowing approximate walking distances (meters)",
            pad=20,
            size=16,
        )

        # Add colorbar legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for colorbar
        cbar = plt.colorbar(sm, ax=ax)  # Specify ax
        cbar.set_label("Distance (meters)")

        ax.axis("off")
        plt.tight_layout()

        return fig, pos_scaled


essential_distances = [
    # Main path through thrill rides
    ("Roller Coaster", "Giga Coaster", 1500),  # Red force, Dragon Khan
    ("Giga Coaster", "Inverted Coaster", 150),  # Dragon Khan, Shambhala
    ("Inverted Coaster", "Flying Coaster", 900),  # Shambhala, Stampida
    ("Flying Coaster", "Superman Ride", 800),  # Furius Baco
    # Water ride section
    ("Water Ride", "Log Flume", 400),  # Ciclon Tropical, silver river flume
    ("Log Flume", "Rapids Ride", 300),  # silver river flume, El Torente
    ("Rapids Ride", "Roller Coaster", 650),  # El Torente, Red force
    # Drop ride section
    ("Drop Tower", "Free Fall", 450),  # Hurakan Condor, King Khajuna
    ("Free Fall", "Power Tower", 850),  # King Khajuna, Thrill towers
    ("Power Tower", "Vertical Drop", 1000),  # Thrill towers, El Salto de Blas
    ("Vertical Drop", "Giga Coaster", 500),  # El salto de Blas, Dragon Khan
    # Family rides section
    ("Merry Go Round", "Circus Train", 1000),  # carousel, sesmoventura station
    ("Circus Train", "Kiddie Coaster", 100),  # sesmoventura station, tami tami
    ("Kiddie Coaster", "Crazy Bus", 50),  # tami tami, coco piloto
    ("Crazy Bus", "Scooby Doo", 25),  # coco piloto, La Granja De Elmo
    ("Scooby Doo", "Water Ride", 240),  # La Granja De Elmo, tutuki splash
    # Flat rides section
    ("Bumper Cars", "Go-Karts", 850),  # Buffalo rodeo, Maranello Grand race
    ("Go-Karts", "Crazy Dance", 750),  # Maranello grand race, Aloha Tahiti
    ("Crazy Dance", "Tilt-A-Whirl", 450),  # Aloha Tahiti, Tea cups
    ("Tilt-A-Whirl", "Spinning Coaster", 600),  # Tea cups, Volpaiute
    ("Spinning Coaster", "Drop Tower", 220),  # Volpaiute, Hurakan Condor
    # Transportation/Special attractions
    ("Monorail", "Skyway", 400),  # coco piloto, furius baco
    ("Skyway", "Gondola", 1000),  # furius baco, hurakan condor (beside)
    ("Gondola", "Zipline", 10),  # hurakan condor (beside), beside
    ("Zipline", "Bungee Jump", 300),
    ("Bungee Jump", "Sling Shot", 300),  # hard to find sling shot and bungee
    # Cross-connections for spatial accuracy
    ("Merry Go Round", "Bumper Cars", 350),  # carousel, buffalo rodeo
    ("Water Ride", "Monorail", 1000),  # ciclon tropical, coco piloto
    ("Superman Ride", "Sling Shot", 800),  # Furius Baco, hurakan condor
    ("Rapids Ride", "Gondola", 600),  # El Torente, hurakan condor
    ("Spinning Coaster", "Flying Coaster", 800),  # Volpaiute, Furius Baco
]


PortAventura_attractions = list(
    link_attractions_df[link_attractions_df["PARK"] == "PortAventura World"][
        "ATTRACTION"
    ]
)

PortAventura_park = ThemeParkGraph(PortAventura_attractions)

for attr1, attr2, distance in essential_distances:
    if attr1 and attr2 in PortAventura_attractions:  # Exclude Tivoli Gardens
        PortAventura_park.add_manual_distance(attr1, attr2, distance)

PortAventura_park.generate_paths()
PortAventura_park.visualize(
    use_distance_layout=True
)  # Set to True for distance-based layout

# Streamlit app
st.title("Euro-Park Dynamic Scheduling")
st.write(
    "Welcome to Euro-Park! Select your preferences and visit duration to get an optimized schedule."
)

# User inputs
# Sidebar for user preferences
st.sidebar.header("User Preferences")

# Add multiselect for attractions preferences
preferences = st.sidebar.multiselect(
    "Select your top 3 preferences",
    PortAventura_attractions,
    default=["Circus Train", "Crazy Dance", "Dizzy Dropper"],
    key="preferences_multiselect",  # Unique key for the multiselect widget
)

# Add sliders for entry and exit times
entry_time = st.sidebar.slider(
    "Entry Time",
    10,
    16,
    10,
    key="entry_time_slider",  # Unique key for the slider widget
)

exit_time = st.sidebar.slider(
    "Exit Time",
    12,
    19,
    16,
    key="exit_time_slider",  # Unique key for the slider widget
)

# Add date input for Date of Visit
date_of_visit = st.sidebar.date_input(
    "Date of Visit",
    datetime.datetime(2022, 5, 22),  # Default value: 1st January 2018
    key="date_of_visit_input",  # Unique key for the date input widget
)

# Display the selected inputs
st.sidebar.write(f"Selected preferences: {preferences}")
st.sidebar.write(f"Entry Time: {entry_time} o'clock")
st.sidebar.write(f"Exit Time: {exit_time} o'clock")
st.sidebar.write(f"Date of Visit: {date_of_visit}")


st.write("Synthetic User Data", synthetic_users[0:500])


@st.cache_resource
def get_cached_figure(_park):
    fig, pos_scaled = _park.visualize(use_distance_layout=True)
    return fig, pos_scaled


fig, pos_scaled = get_cached_figure(PortAventura_park)

# Display the plot in Streamlit
st.pyplot(fig)


def predict_wait_time(attraction, arrival_time, waiting_df):
    """
    Estimate the wait time for an attraction by finding the lowest available wait time
    within a reasonable time window instead of just picking the closest.
    """
    wait_times = waiting_df[waiting_df["ENTITY_DESCRIPTION_SHORT"] == attraction]
    wait_times = wait_times.sort_values(by="DEB_TIME")

    # Filter times within a 60-minute window
    relevant_times = wait_times[
        (wait_times["DEB_TIME_HOUR"] >= arrival_time)
        & (wait_times["DEB_TIME_HOUR"] <= arrival_time + 1)
    ]

    if not relevant_times.empty:
        min_wait_time = relevant_times["WAIT_TIME_MAX"].min()
        return (
            min_wait_time if min_wait_time > 0 else 2
        )  # Ensure a default of 2 minutes
    else:
        return 2  # Default fallback


def get_ride_duration(attraction, waiting_df):
    ride_time = waiting_df[waiting_df["ENTITY_DESCRIPTION_SHORT"] == attraction][
        "UP_TIME"
    ]
    return (
        ride_time.values[0] if not ride_time.values[0] == 0 else 5
    )  # Default to 5 min


def get_travel_time(attraction1, attraction2, theme_park):
    """
    Get travel time between two attractions.
    """
    distance = theme_park.get_distance(attraction1, attraction2)
    travel_time = distance / 100
    return travel_time if travel_time >= 1 else 1


def get_crowd_penalty(attraction, overlapping_visitors):
    """
    Compute a penalty based on the number of other visitors likely to visit this attraction.
    """
    return overlapping_visitors[
        (overlapping_visitors["ride_preference_1"] == attraction)
        | (overlapping_visitors["ride_preference_2"] == attraction)
        | (overlapping_visitors["ride_preference_3"] == attraction)
    ].shape[0]


def compute_score(wait_time, distance, crowd_penalty, itinerary, user_preferences):
    """
    Compute a weighted score balancing waiting time, distance, crowd management, and user preferences.
    """
    w_wait = -1.5  # Negative weight for wait time (lower is better)
    w_distance = -0.2  # Negative weight for distance (lower is better)
    w_crowd = -2  # Higher penalty for crowded attractions
    w_pref = 3  # Positive weight for visiting top preferences first

    # Calculate preference score (higher if preferred attractions appear earlier in the list)
    preference_score = 0
    for index, attraction in enumerate(itinerary):
        if attraction == user_preferences[0]:  # First choice, highest boost
            preference_score += 3
        elif attraction == user_preferences[1]:  # Second choice, medium boost
            preference_score += 2
        elif attraction == user_preferences[2]:  # Third choice, lowest boost
            preference_score += 1

    return (
        w_wait * wait_time
        + w_distance * distance
        + w_crowd * crowd_penalty
        + w_pref * preference_score
    )


def evaluate_itinerary(
    itinerary,
    start_time,
    waiting_df,
    date,
    overlapping_visitors,
    theme_park,
    user_preferences,
    end_time,
):
    # Isolate the date of interest
    waiting_df = waiting_df[pd.to_datetime(waiting_df["WORK_DATE"]).dt.date == date]

    current_time = start_time
    total_wait_time = 0
    total_distance = 0
    total_crowd_penalty = 0
    schedule = []

    for i, attraction in enumerate(itinerary):
        # times for each attraction
        predicted_wait_time = predict_wait_time(attraction, current_time, waiting_df)
        ride_duration = get_ride_duration(attraction, waiting_df)
        crowd_penalty = get_crowd_penalty(attraction, overlapping_visitors)
        travel_time = (
            get_travel_time(itinerary[i - 1], attraction, theme_park) if i > 0 else 0
        )

        total_wait_time += predicted_wait_time
        total_distance += travel_time
        total_crowd_penalty += crowd_penalty

        departure_time = current_time + (predicted_wait_time + ride_duration) / 60
        next_travel_time = (
            get_travel_time(attraction, itinerary[i + 1], theme_park)
            if i < len(itinerary) - 1
            else 0
        )

        schedule.append(
            {
                "attraction": attraction,
                "arrival_time": round(current_time, 2),
                "wait_time": predicted_wait_time,
                "ride_duration": ride_duration,
                "departure_time": round(departure_time, 2),
                "travel_time_to_next": math.ceil(next_travel_time),
            }
        )

        current_time += (
            predicted_wait_time + ride_duration + math.ceil(travel_time)
        ) / 60

    # Fill remaining time with low-wait attractions
    schedule = fill_gaps(schedule, current_time, end_time, waiting_df, theme_park)

    score = compute_score(
        total_wait_time,
        total_distance,
        total_crowd_penalty,
        itinerary,
        user_preferences,
    )

    return schedule, score


def fill_gaps(schedule, current_time, end_time, waiting_df, theme_park):
    available_time = end_time - current_time

    if available_time < 0.25:  # Ignore tiny gaps
        return schedule

    low_wait_attractions = get_low_wait_time_attractions(
        waiting_df, current_time, available_time
    )

    for attraction in low_wait_attractions:
        wait_time = predict_wait_time(attraction, current_time, waiting_df)
        ride_duration = get_ride_duration(attraction, waiting_df)
        travel_time = (
            get_travel_time(schedule[-1]["attraction"], attraction, theme_park)
            if schedule
            else 0
        )

        if current_time + (wait_time + ride_duration + travel_time) / 60 > end_time:
            break  # Don't overfill!

        departure_time = current_time + (wait_time + ride_duration) / 60
        schedule.append(
            {
                "attraction": attraction,
                "arrival_time": round(current_time, 2),
                "wait_time": wait_time,
                "ride_duration": ride_duration,
                "departure_time": round(departure_time, 2),
                "travel_time_to_next": 0,
            }
        )
        current_time = departure_time

    return schedule


def get_low_wait_time_attractions(waiting_df, current_time, available_time):
    filtered = waiting_df[
        (waiting_df["DEB_TIME_HOUR"] >= current_time)
        & (waiting_df["WAIT_TIME_MAX"] < 10)
    ]
    # st.write(filtered) # for debugging
    return filtered.sort_values("WAIT_TIME_MAX")["ENTITY_DESCRIPTION_SHORT"].tolist()


def optimize_schedule(new_visitor, synthetic_users, theme_park, waiting_df, date):
    overlapping_visitors = synthetic_users[
        (synthetic_users["entry_time"] <= new_visitor["exit_time"])
        & (synthetic_users["exit_time"] >= new_visitor["entry_time"])
    ]

    # Sort preferences by historical lowest wait times
    sorted_preferences = sorted(
        new_visitor["preferences"],
        key=lambda ride: predict_wait_time(ride, new_visitor["entry_time"], waiting_df),
    )

    attraction_permutations = list(itertools.permutations(sorted_preferences))
    best_schedule, best_score = None, float("-inf")

    for itinerary in attraction_permutations:
        schedule, score = evaluate_itinerary(
            itinerary,
            new_visitor["entry_time"],
            waiting_df,
            date,
            overlapping_visitors,
            theme_park,
            new_visitor["preferences"],
            new_visitor["exit_time"],
        )

        if score > best_score:
            best_schedule, best_score = schedule, score

    return best_schedule


########################################
# NEED TO UPDATE THIS FOR PREDICTIONS

# Ensure 'WORK_DATE' column is in datetime format
waiting_df["WORK_DATE"] = pd.to_datetime(waiting_df["WORK_DATE"])

new_visitor = {
    "preferences": [preferences[0], preferences[1], preferences[2]]
    if len(preferences) >= 3
    else preferences,
    "entry_time": entry_time,
    "exit_time": exit_time,
}


# Run optimization
# Using inputs
if len(preferences) == 3:
    generate_schedule = True
if generate_schedule:
    optimized_schedule = optimize_schedule(
        new_visitor, synthetic_users, PortAventura_park, waiting_df, date_of_visit
    )

    # st.markdown("## This is a schedule for the inputs you've given")
    # st.write(optimized_schedule)  # we can get rid of this

    st.markdown("## 🎢 Your Optimized Park Schedule")

    for idx, ride in enumerate(optimized_schedule):
        # Round travel time up
        travel_time = math.ceil(ride["travel_time_to_next"])

        # Convert decimal times to readable HHhMM format
        arrival_time_h = int(ride["arrival_time"])
        arrival_time_m = round((ride["arrival_time"] - arrival_time_h) * 60)
        departure_time_h = int(ride["departure_time"])
        departure_time_m = round((ride["departure_time"] - departure_time_h) * 60)

        arrival_str = f"{arrival_time_h}h{arrival_time_m:02d}"
        departure_str = f"{departure_time_h}h{departure_time_m:02d}"

        # Display ride details
        st.markdown(f"### ⏰ {arrival_str} - **{ride['attraction']}**")
        st.write(f"- Estimated waiting time: Usually ~ {ride['wait_time']} mins")
        st.write(f"- Ride duration: {ride['ride_duration']} mins")
        st.write(f"- Travel time to next attraction: {travel_time} min")
        st.write(f"- Departure time: {departure_str}")
        st.markdown("---")

    st.markdown("Enjoy your day at the park! 🎡🎠")

    ## SAMPLING
    st.markdown("## Now showing how the paths of 500 random people will interact")
    st.markdown("#### Bubbles are scaled relative to capacity at that attraction")

    sampled_visitors = synthetic_users.sample(n=500, replace=True).reset_index(
        drop=True
    )

    sampled_visitors_dict = sampled_visitors.to_dict(orient="records")

    capacity_dict = dict(
        zip(ride_capacity["ENTITY_DESCRIPTION_SHORT"], ride_capacity["CAPACITY"])
    )

    # Function to convert decimal hours to "HH:MM" format
    def decimal_to_time(decimal_time):
        hours = int(decimal_time)
        minutes = int((decimal_time - hours) * 60)
        return f"{hours:02}:{minutes:02}"  # Format as HH:MM

    PortAventura_attractions = list(
        link_attractions_df[link_attractions_df["PARK"] == "PortAventura World"][
            "ATTRACTION"
        ]
    )
    PortAventura_park = ThemeParkGraph(PortAventura_attractions)

    # Add manual distances
    for attr1, attr2, distance in essential_distances:
        if attr1 and attr2 in PortAventura_attractions:  # Exclude Tivoli Gardens
            PortAventura_park.add_manual_distance(attr1, attr2, distance)

    # Now you can use pos for your animation
    attractions = list(ride_capacity["ENTITY_DESCRIPTION_SHORT"].unique())

    x = [pos_scaled[node][0] for node in attractions]
    y = [pos_scaled[node][1] for node in attractions]

    # Create edge trace (static, no labels or numbers)
    edge_trace = go.Scatter(
        x=sum(
            [
                [pos_scaled[edge[0]][0], pos_scaled[edge[1]][0], None]
                for edge in PortAventura_park.G.edges
            ],
            [],
        ),
        y=sum(
            [
                [pos_scaled[edge[0]][1], pos_scaled[edge[1]][1], None]
                for edge in PortAventura_park.G.edges
            ],
            [],
        ),
        mode="lines",
        line=dict(width=1, color="gray"),
        hoverinfo="none",  # Prevents hover text on edges
    )

    # Generate frames for animation
    frames = []
    time_intervals = np.arange(10, 19, 0.25)

    for time in time_intervals:
        # Count visitors at each attraction at this time
        active_visitors = schedule_df[
            (schedule_df["arrival_time"] <= time)
            & (schedule_df["departure_time"] >= time)
        ]

        node_sizes = []
        node_labels = []
        for node in attractions:
            visitor_count = active_visitors[
                active_visitors["attraction"] == node
            ].shape[0]
            normalized_size = (
                visitor_count / capacity_dict[node]
            )  # Normalize by capacity
            scaled_size = normalized_size * 100  # Scale the size
            node_sizes.append(scaled_size)
            node_labels.append(
                f"{node}: {visitor_count}"
            )  # Only append labels to nodes

        formatted_time = decimal_to_time(time)

        # Frame only updates node positions and labels, NOT edges
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=[pos_scaled[node][0] for node in attractions],
                        y=[pos_scaled[node][1] for node in attractions],
                        mode="markers+text",
                        marker=dict(size=node_sizes, color="blue", opacity=0.7),
                        text=node_labels,  # Ensure only nodes have labels
                        textposition="top center",
                        hoverinfo="text",
                    )
                ],
                name=formatted_time,  # Use formatted time
            )
        )

    # Initialize figure with only edge trace (nodes appear via animation)
    fig = go.Figure(
        data=[edge_trace],  # Keep only edges static initially
        layout=go.Layout(
            title="Visitor Distribution Over Time",
            width=1200,
            height=800,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {"prefix": "Time: ", "font": {"size": 20}},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": -0.1,
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": f.name,
                            "method": "animate",
                        }
                        for f in frames
                    ],
                }
            ],
        ),
        frames=frames,  # Ensure frames only update nodes
    )

    st.plotly_chart(fig)

# def optimize_schedule(new_visitor, synthetic_users, theme_park, waiting_df):
#     """
#     Optimize a visitor's itinerary based on predicted wait times, distance, and crowd management.
#     """
#     overlapping_visitors = synthetic_users[
#         (synthetic_users["entry_time"] <= new_visitor["exit_time"])
#         & (synthetic_users["exit_time"] >= new_visitor["entry_time"])
#     ]

#     # Generate all permutations of the preferred attractions
#     attraction_permutations = list(itertools.permutations(new_visitor["preferences"]))

#     best_schedule = None
#     best_score = float('-inf')

#     for itinerary in attraction_permutations:
#         schedule, score = evaluate_itinerary(itinerary, new_visitor["entry_time"], waiting_df, overlapping_visitors, theme_park, new_visitor["preferences"])

#         if score > best_score:
#             best_schedule = schedule
#             best_score = score

#     return best_schedule

# def evaluate_itinerary(itinerary, start_time, waiting_df, overlapping_visitors, theme_park, user_preferences):
#     """
#     Evaluate a given itinerary based on predicted wait times, travel distance, and visitor load.
#     """
#     current_time = start_time
#     total_wait_time = 0
#     total_distance = 0
#     total_crowd_penalty = 0
#     schedule = []

#     for i, attraction in enumerate(itinerary):
#         # Predict wait time based on arrival time
#         predicted_wait_time = predict_wait_time(attraction, current_time, waiting_df)

#         # Get ride duration
#         ride_duration = get_ride_duration(attraction, waiting_df)

#         # Compute crowd penalty (number of overlapping visitors also likely to visit this attraction)
#         crowd_penalty = get_crowd_penalty(attraction, overlapping_visitors)

#         # Compute distance to next attraction
#         if i > 0:
#             travel_time = get_travel_time(itinerary[i-1], attraction, theme_park)
#         else:
#             travel_time = 0

#         # Update total metrics
#         total_wait_time += predicted_wait_time
#         total_distance += travel_time
#         total_crowd_penalty += crowd_penalty

#         # Add to the schedule
#         next_travel_time = get_travel_time(attraction, itinerary[i + 1], theme_park) if i < len(itinerary) - 1 else 0
#         departure_time = current_time + (predicted_wait_time + ride_duration) / 60

#         schedule.append({
#             "attraction": attraction,
#             "arrival_time": round(current_time, 2),
#             "wait_time": predicted_wait_time,
#             "ride_duration": ride_duration,
#             "departure_time": round(departure_time, 2),
#             "travel_time_to_next": math.ceil(next_travel_time)  # Assign travel time to next attraction
#         })

#         # Update current time
#         current_time += (predicted_wait_time + ride_duration + math.ceil(travel_time)) / 60  # Convert minutes to hours

#     # Compute overall score using a weighted approach (Pareto-inspired)

#     score = compute_score(total_wait_time, total_distance, total_crowd_penalty, itinerary, list(new_visitor["preferences"]))

#     return schedule, score

# def predict_wait_time(attraction, arrival_time, waiting_df):
#     """
#     Estimate the wait time for an attraction based on the visitor's arrival time.
#     """
#     wait_times = waiting_df[(waiting_df["ENTITY_DESCRIPTION_SHORT"] == attraction)]
#     wait_times = wait_times.sort_values(by="DEB_TIME")

#     closest_time = wait_times.iloc[(wait_times["DEB_TIME_HOUR"] - arrival_time).abs().argsort()[:1]]
#     return closest_time["WAIT_TIME_MAX"].values[0] if not closest_time["WAIT_TIME_MAX"].values[0] == 0 else 2  # Default to 10 min

# def get_ride_duration(attraction, waiting_df):
#     ride_time = waiting_df[waiting_df["ENTITY_DESCRIPTION_SHORT"] == attraction]["UP_TIME"]
#     return ride_time.values[0] if not ride_time.values[0] == 0 else 5  # Default to 5 min

# def get_travel_time(attraction1, attraction2, theme_park):
#     """
#     Get travel time between two attractions.
#     """
#     distance = theme_park.get_distance(attraction1, attraction2)
#     travel_time = distance / 100
#     return travel_time if travel_time >=1 else 1

# def get_crowd_penalty(attraction, overlapping_visitors):
#     """
#     Compute a penalty based on the number of other visitors likely to visit this attraction.
#     """
#     return overlapping_visitors[
#         (overlapping_visitors["ride_preference_1"] == attraction) |
#         (overlapping_visitors["ride_preference_2"] == attraction) |
#         (overlapping_visitors["ride_preference_3"] == attraction)
#     ].shape[0]

# def compute_score(wait_time, distance, crowd_penalty, itinerary, user_preferences):
#     """
#     Compute a weighted score balancing waiting time, distance, crowd management, and user preferences.
#     """
#     w_wait = -1       # Negative weight for wait time (lower is better)
#     w_distance = -0.5 # Negative weight for distance (lower is better)
#     w_crowd = -2      # Higher penalty for crowded attractions
#     w_pref = 3        # Positive weight for visiting top preferences first

#     # Calculate preference score (higher if preferred attractions appear earlier in the list)
#     preference_score = 0
#     for index, attraction in enumerate(itinerary):
#         if attraction == user_preferences[0]:  # First choice, highest boost
#             preference_score += 3
#         elif attraction == user_preferences[1]:  # Second choice, medium boost
#             preference_score += 2
#         elif attraction == user_preferences[2]:  # Third choice, lowest boost
#             preference_score += 1

#     return (
#         w_wait * wait_time +
#         w_distance * distance +
#         w_crowd * crowd_penalty +
#         w_pref * preference_score
#     )


# # Streamlit app
# st.title("Amusement Park Dynamic Scheduling")
# st.write(
#     "Welcome to Portaventura World! Select your preferences and visit duration to get an optimized schedule."
# )

# # User inputs
# st.sidebar.header("User Preferences")
# preferences = st.sidebar.multiselect(
#     "Select your top 3 preferences",
#     attractions,
#     default=attractions[:3],
#     key="preferences_multiselect",  # Unique key for the multiselect widget
# )
# entry_time = st.sidebar.slider(
#     "Entry Time",
#     10,
#     19,
#     10,
#     key="entry_time_slider",  # Unique key for the slider widget
# )
# exit_time = st.sidebar.slider(
#     "Exit Time",
#     10,
#     19,
#     19,
#     key="exit_time_slider",  # Unique key for the slider widget
# )

# st.write("Synthetic User Data", synthetic_users)

# new_visitor = {
#     "preferences": [preferences[0], preferences[1], preferences[2]] if len(preferences) >= 3 else preferences,
#     "entry_time": entry_time,
#     "exit_time": exit_time,
# }

# optimized_schedule = optimize_schedule(new_visitor, synthetic_users, theme_park, waiting_df)

# # Show optimized schedule for the new visitor
# st.subheader("Optimized Schedule for You")
# st.write("Your optimized schedule based on your preferences and entry/exit times:")

# for stop in optimized_schedule:
#     arrival_time_formatted = f"{int(stop['arrival_time'])}h{int((stop['arrival_time'] % 1) * 60):02d}"
#     departure_time_formatted = f"{int(stop['departure_time'])}h{int((stop['departure_time'] % 1) * 60):02d}"

#     st.markdown(f"### ⏰ {arrival_time_formatted} - {stop['attraction']}")
#     st.write(f"**- Estimated Waiting Time:** {stop['wait_time']} min")
#     st.write(f"**- Ride Duration:** {stop['ride_duration']} min")
#     st.write(f"**- Travel Time to Next Attraction:** {round(stop['travel_time_to_next'], 2)} min")
#     st.write(f"**- Departure Time:** {departure_time_formatted}")
#     st.write('---')

# # Visualize Ride Activity
# def visualize_activity(ride_data, theme_park):
#     # Visualize the map with capacity at selected time
#     fig = theme_park.visualize(ride_data)
#     st.plotly_chart(fig)


# # Display the interactive map
# visualize_activity(ride_capacity, theme_park)

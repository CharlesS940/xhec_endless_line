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


@st.cache_data
def load_predictions():
    return pd.read_csv(
        "data/waiting_time_predictions.csv"
    )  # Our predictions of waiting times


synthetic_users = load_data()
ride_capacity = load_ride_data()
waiting_df = load_waiting_data()
schedule_df = load_pathing_schedules()
link_attractions_df = load_attractions_location()
predictions_df = load_predictions()

# Manipulating the predictions_df so that it will work

predictions_df["DEB_TIME"] = pd.to_datetime(predictions_df["DEB_TIME"])
predictions_df["WORK_DATE"] = predictions_df["DEB_TIME"].dt.date
predictions_df["DEB_TIME_HOUR"] = predictions_df["DEB_TIME"].dt.hour

predictions_df.rename(columns={"ride": "ENTITY_DESCRIPTION_SHORT"}, inplace=True)
predictions_df.rename(columns={"pred": "WAIT_TIME_MAX"}, inplace=True)

# add in average up time for that ride
predictions_df = predictions_df.merge(
    ride_capacity[["ENTITY_DESCRIPTION_SHORT", "UP_TIME"]],
    on="ENTITY_DESCRIPTION_SHORT",
    how="left",
)
# round the up time
predictions_df["UP_TIME"] = predictions_df[
    "UP_TIME"
].round()  # Round to the nearest integer

waiting_df = waiting_df[
    [
        "DEB_TIME",
        "ENTITY_DESCRIPTION_SHORT",
        "WAIT_TIME_MAX",
        "WORK_DATE",
        "DEB_TIME_HOUR",
        "UP_TIME",
    ]
]

# Concatenate both the predictions and waiting times to allow choice over any date in the range
waiting_df = pd.concat([waiting_df, predictions_df], ignore_index=True)


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
    datetime.datetime(2022, 6, 20),  # Default value: 20th June 2022
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
    Predicts wait time using interpolation for more accurate results.
    """
    wait_times = waiting_df[waiting_df["ENTITY_DESCRIPTION_SHORT"] == attraction].copy()

    if wait_times.empty:
        return 2  # Default fallback if no data available

    wait_times["DEB_TIME"] = pd.to_datetime(wait_times["DEB_TIME"], errors="coerce")
    wait_times = wait_times.sort_values("DEB_TIME")

    # Get closest past & future times
    before = wait_times[wait_times["DEB_TIME_HOUR"] <= arrival_time]
    after = wait_times[wait_times["DEB_TIME_HOUR"] > arrival_time]

    if not before.empty and not after.empty:
        # Interpolation
        t1, w1 = before.iloc[-1][["DEB_TIME_HOUR", "WAIT_TIME_MAX"]]
        t2, w2 = after.iloc[0][["DEB_TIME_HOUR", "WAIT_TIME_MAX"]]
        return max(2, w1 + (arrival_time - t1) * (w2 - w1) / (t2 - t1))

    elif not before.empty:
        return max(2, before["WAIT_TIME_MAX"].iloc[-1])  # Use last known value
    elif not after.empty:
        return max(2, after["WAIT_TIME_MAX"].iloc[0])  # Use next known value

    return 2  # Default fallback


def get_ride_duration(attraction, waiting_df):
    """
    Fetches ride duration, defaulting to 5 minutes if unknown or invalid.
    """
    # Fetch the ride duration
    ride_time = waiting_df.loc[
        waiting_df["ENTITY_DESCRIPTION_SHORT"] == attraction, "UP_TIME"
    ]

    # Check if the ride time is valid (not empty, not zero, and not NaN)
    if not ride_time.empty and pd.notna(ride_time.iloc[0]) and ride_time.iloc[0] > 0:
        return ride_time.iloc[0]
    else:
        # Return 5 minutes if the ride time is zero, NaN, or missing
        return 10


def get_travel_time(attraction1, attraction2, theme_park):
    """
    Gets travel time between two attractions.
    """
    if attraction1 == attraction2:
        return 0  # No travel needed
    distance = theme_park.get_distance(attraction1, attraction2)
    return max(1, distance / 100)


def evaluate_itinerary(itinerary, start_time, waiting_df, date, theme_park, end_time):
    """
    Evaluates an itinerary and returns the schedule.
    """
    waiting_df = waiting_df[pd.to_datetime(waiting_df["WORK_DATE"]).dt.date == date]
    schedule, current_time = [], start_time

    for i, attraction in enumerate(itinerary):
        wait_time = predict_wait_time(attraction, current_time, waiting_df)
        ride_duration = get_ride_duration(attraction, waiting_df)
        travel_time = (
            get_travel_time(itinerary[i - 1], attraction, theme_park) if i > 0 else 0
        )

        if current_time + (wait_time + ride_duration + travel_time) / 60 > end_time:
            break  # Stop if we exceed available time

        departure_time = current_time + (wait_time + ride_duration) / 60
        schedule.append(
            {
                "attraction": attraction,
                "arrival_time": round(current_time, 2),
                "wait_time": wait_time,
                "ride_duration": ride_duration,
                "departure_time": round(departure_time, 2),
                "travel_time_to_next": math.ceil(travel_time),
            }
        )

        current_time = departure_time + travel_time / 60  # Move to the next attraction

    return schedule


def optimize_schedule(new_visitor, theme_park, waiting_df, date):
    """
    Optimizes the best possible schedule based on visitor preferences.
    """
    sorted_preferences = sorted(
        new_visitor["preferences"],
        key=lambda ride: predict_wait_time(ride, new_visitor["entry_time"], waiting_df),
    )

    best_schedule, best_score = None, float("-inf")

    for itinerary in itertools.permutations(set(sorted_preferences)):
        schedule = evaluate_itinerary(
            itinerary,
            new_visitor["entry_time"],
            waiting_df,
            date,
            theme_park,
            new_visitor["exit_time"],
        )

        # Scoring based on minimizing wait & travel time
        total_wait = sum(item["wait_time"] for item in schedule)
        total_travel = sum(item["travel_time_to_next"] for item in schedule)
        score = -1.5 * total_wait - 0.5 * total_travel  # Higher score is better

        if score > best_score:
            best_schedule, best_score = schedule, score

    return best_schedule


def get_low_wait_time_attractions(waiting_df, current_time, available_time):
    filtered = waiting_df[
        (waiting_df["DEB_TIME_HOUR"] >= current_time)
        & (waiting_df["WAIT_TIME_MAX"] < 10)
    ]
    # st.write(filtered) # for debugging
    return filtered.sort_values("WAIT_TIME_MAX")["ENTITY_DESCRIPTION_SHORT"].tolist()


def evaluate_itinerary(itinerary, start_time, waiting_df, date, theme_park, end_time):
    """
    Evaluates an itinerary and returns the schedule, filling available gaps if possible.
    """
    waiting_df = waiting_df[pd.to_datetime(waiting_df["WORK_DATE"]).dt.date == date]
    schedule, current_time = [], start_time

    for i, attraction in enumerate(itinerary):
        wait_time = predict_wait_time(attraction, current_time, waiting_df)
        ride_duration = get_ride_duration(attraction, waiting_df)
        travel_time = (
            get_travel_time(itinerary[i - 1], attraction, theme_park) if i > 0 else 0
        )

        if current_time + (wait_time + ride_duration + travel_time) / 60 > end_time:
            break  # Stop if we exceed available time

        departure_time = current_time + (wait_time + ride_duration) / 60
        schedule.append(
            {
                "attraction": attraction,
                "arrival_time": round(current_time, 2),
                "wait_time": wait_time,
                "ride_duration": ride_duration,
                "departure_time": round(departure_time, 2),
                "travel_time_to_next": math.ceil(travel_time),
            }
        )

        current_time = departure_time + travel_time / 60  # Move to the next attraction

    # Fill remaining time with low-wait attractions
    # Fill remaining time with low-wait attractions
    remaining_time = end_time - current_time
    if remaining_time > 0:
        # Look for attractions that fit within the remaining time
        low_wait_attractions = get_low_wait_time_attractions(
            waiting_df, current_time, remaining_time
        )

        # Inside evaluate_itinerary function, in the gap-filling section:
        visited_attractions = {
            s["attraction"] for s in schedule
        }  # Track visited attractions
        for attraction in low_wait_attractions:
            if attraction in visited_attractions:
                continue  # Skip if already visited
            if schedule:  # If there are previous attractions
                travel_time = get_travel_time(
                    schedule[-1]["attraction"], attraction, theme_park
                )
            else:
                travel_time = 0

            wait_time = predict_wait_time(attraction, current_time, waiting_df)
            ride_duration = get_ride_duration(attraction, waiting_df)

            # Check if this attraction fits in the remaining time
            if (
                current_time + (wait_time + ride_duration + travel_time) / 60
                <= end_time
            ):
                departure_time = current_time + (wait_time + ride_duration) / 60
                schedule.append(
                    {
                        "attraction": attraction,
                        "arrival_time": round(current_time, 2),
                        "wait_time": wait_time,
                        "ride_duration": ride_duration,
                        "departure_time": round(departure_time, 2),
                        "travel_time_to_next": math.ceil(travel_time),
                    }
                )
                visited_attractions.add(attraction)  # Mark as visited
                current_time = departure_time + travel_time / 60

    return schedule


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
    optimized_schedule = optimized_schedule = optimize_schedule(
        new_visitor, PortAventura_park, waiting_df, date_of_visit
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
        st.write(
            f"- Estimated waiting time: Usually ~ {math.ceil(ride['wait_time'])} mins"
        )
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

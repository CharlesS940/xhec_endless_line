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


@st.cache_data
def load_data():
    return pd.read_csv("data/synthetic_user_data.csv")


@st.cache_data
def load_ride_data():
    return pd.read_csv("data/ride_capacity.csv")

@st.cache_data
def load_waiting_data():
    return pd.read_csv("data/waiting_times.csv")

synthetic_users = load_data()
ride_capacity = load_ride_data()
waiting_df = load_waiting_data()


# Initialize Graph
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

        initial_pos = np.random.rand(len(self.G.nodes()), 2)
        result = minimize(stress, initial_pos.flatten(), method="L-BFGS-B")
        final_pos = result.x.reshape(-1, 2)

        return {node: pos for node, pos in zip(self.G.nodes(), final_pos)}

    def visualize(self, capacity_data, time_slider_value=10):
        pos = self.stress_majorization_layout()

        # Extract node positions
        x_vals = [pos[node][0] for node in self.G.nodes()]
        y_vals = [pos[node][1] for node in self.G.nodes()]

        # Adjust node sizes based on capacity and time slider
        node_sizes = [
            capacity_data.get(node, {}).get(time_slider_value, 0) * 1000
            for node in self.G.nodes()
        ]

        # Create edges with faint lines
        edge_x = []
        edge_y = []
        for u, v in self.G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_y.append(y0)
            edge_y.append(y1)

        # Create the plot
        fig = go.Figure()

        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="lightgray", width=0.5),
                hoverinfo="none",
            )
        )

        # Add nodes (rides)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color="lightblue",
                    opacity=0.7,
                    line=dict(width=1, color="black"),
                ),
                text=[
                    f"{node}<br>{int(capacity_data.get(node, {}).get(time_slider_value, 0) * 100)}% capacity"
                    for node in self.G.nodes()
                ],
                textposition="middle center",
                hoverinfo="text",
            )
        )

        fig.update_layout(
            title="Interactive Theme Park Map with Capacity Utilization",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            plot_bgcolor="white",
            hovermode="closest",
            margin=dict(t=40, b=40, l=40, r=40),
            height=800,
            width=800,
            sliders=[
                {
                    "currentvalue": {
                        "prefix": "Time: ",
                        "visible": True,
                        "xanchor": "center",
                    },
                    "pad": {"b": 10},
                    "len": 0.9,
                    "x": 0.5,
                    "y": 0.02,
                    "steps": [
                        {
                            "args": [
                                [time],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": f"{time}h",
                            "method": "animate",
                        }
                        for time in range(10, 19)
                    ],
                }
            ],
        )

        return fig


# Attraction list
attractions = [
    "Bumper Cars",
    "Bungee Jump",
    "Circus Train",
    "Crazy Dance",
    "Dizzy Dropper",
    "Drop Tower",
    "Flying Coaster",
    "Free Fall",
    "Giant Wheel",
    "Giga Coaster",
    "Go-Karts",
    "Haunted House",
    "Himalaya Ride",
    "Inverted Coaster",
    "Kiddie Coaster",
    "Merry Go Round",
    "Oz Theatre",
    "Rapids Ride",
    "Roller Coaster",
    "Spinning Coaster",
    "Spiral Slide",
    "Superman Ride",
    "Swing Ride",
    "Vertical Drop",
    "Water Ride",
    "Zipline",
]

theme_park = ThemeParkGraph(attractions)

essential_distances = [
    # Main path through thrill rides
    ("Roller Coaster", "Giga Coaster", 1500),  # Red force, Dragon Khan
    ("Giga Coaster", "Inverted Coaster", 150),  # Dragon Khan, Shambhala
    ("Inverted Coaster", "Flying Coaster", 900),  # Shambhala, Stampida
    ("Flying Coaster", "Superman Ride", 800),  # Furius Baco
    # Water ride section
    ("Water Ride", "Rapids Ride", 400),  # Ciclon Tropical, silver river flume
    # ("Log Flume", "Rapids Ride", 300),  # silver river flume, El Torente
    ("Rapids Ride", "Roller Coaster", 650),  # El Torente, Red force
    # Drop ride section
    ("Drop Tower", "Free Fall", 450),  # Hurakan Condor, King Khajuna
    # ("Free Fall", "Power Tower", 850),  # King Khajuna, Thrill towers
    ("Free Fall", "Vertical Drop", 1000),  # Thrill towers, El Salto de Blas
    ("Vertical Drop", "Giga Coaster", 500),  # El salto de Blas, Dragon Khan
    # Family rides section
    ("Merry Go Round", "Circus Train", 1000),  # carousel, sesmoventura station
    ("Circus Train", "Kiddie Coaster", 100),  # sesmoventura station, tami tami
    ("Kiddie Coaster", "Water Ride", 240),  # tami tami, coco piloto
    # ("Crazy Bus", "Scooby Doo", 25),  # coco piloto, La Granja De Elmo
    # ("Scooby Doo", "Water Ride", 240),  # La Granja De Elmo, tutuki splash
    # Flat rides section
    ("Bumper Cars", "Go-Karts", 850),  # Buffalo rodeo, Maranello Grand race
    ("Go-Karts", "Crazy Dance", 750),  # Maranello grand race, Aloha Tahiti
    ("Crazy Dance", "Spinning Coaster", 450),  # Aloha Tahiti, Tea cups
    # ("Tilt-A-Whirl", "Spinning Coaster", 600),  # Tea cups, Volpaiute
    ("Spinning Coaster", "Drop Tower", 220),  # Volpaiute, Hurakan Condor
    # Transportation/Special attractions
    # ("Monorail", "Skyway", 400),  # coco piloto, furius baco
    # ("Skyway", "Gondola", 1000),  # furius baco, hurakan condor (beside)
    # ("Gondola", "Zipline", 10),  # hurakan condor (beside), beside
    ("Zipline", "Bungee Jump", 300),
    ("Bungee Jump", "Superman Ride", 800),  # hard to find sling shot and bungee
    # Cross-connections for spatial accuracy
    ("Merry Go Round", "Bumper Cars", 350),  # carousel, buffalo rodeo
    ("Water Ride", "Zipline", 1000),  # ciclon tropical, coco piloto
    # ("Superman Ride", "Sling Shot", 800),  # Furius Baco, hurakan condor
    ("Rapids Ride", "Bungee Jump", 600),  # El Torente, hurakan condor
    ("Spinning Coaster", "Flying Coaster", 800),  # Volpaiute, Furius Baco
]

for attr1, attr2, distance in essential_distances:
    theme_park.add_manual_distance(attr1, attr2, distance)
    

def optimize_schedule(new_visitor, synthetic_users, theme_park, waiting_df):
    """
    Optimize a visitor's itinerary based on predicted wait times, distance, and crowd management.
    """
    overlapping_visitors = synthetic_users[
        (synthetic_users["entry_time"] <= new_visitor["exit_time"])
        & (synthetic_users["exit_time"] >= new_visitor["entry_time"])
    ]
    
    # Generate all permutations of the preferred attractions
    attraction_permutations = list(itertools.permutations(new_visitor["preferences"]))
    
    best_schedule = None
    best_score = float('-inf')
    
    for itinerary in attraction_permutations:
        schedule, score = evaluate_itinerary(itinerary, new_visitor["entry_time"], waiting_df, overlapping_visitors, theme_park, new_visitor["preferences"])
        
        if score > best_score:
            best_schedule = schedule
            best_score = score
    
    return best_schedule

def evaluate_itinerary(itinerary, start_time, waiting_df, overlapping_visitors, theme_park, user_preferences):
    """
    Evaluate a given itinerary based on predicted wait times, travel distance, and visitor load.
    """
    current_time = start_time
    total_wait_time = 0
    total_distance = 0
    total_crowd_penalty = 0
    schedule = []
    
    for i, attraction in enumerate(itinerary):
        # Predict wait time based on arrival time
        predicted_wait_time = predict_wait_time(attraction, current_time, waiting_df)
        
        # Get ride duration
        ride_duration = get_ride_duration(attraction, waiting_df)
        
        # Compute crowd penalty (number of overlapping visitors also likely to visit this attraction)
        crowd_penalty = get_crowd_penalty(attraction, overlapping_visitors)
        
        # Compute distance to next attraction
        if i > 0:
            travel_time = get_travel_time(itinerary[i-1], attraction, theme_park)
        else:
            travel_time = 0
        
        # Update total metrics
        total_wait_time += predicted_wait_time
        total_distance += travel_time
        total_crowd_penalty += crowd_penalty
        
        # Add to the schedule
        next_travel_time = get_travel_time(attraction, itinerary[i + 1], theme_park) if i < len(itinerary) - 1 else 0
        departure_time = current_time + (predicted_wait_time + ride_duration) / 60
        
        schedule.append({
            "attraction": attraction,
            "arrival_time": round(current_time, 2),
            "wait_time": predicted_wait_time,
            "ride_duration": ride_duration,
            "departure_time": round(departure_time, 2),
            "travel_time_to_next": round(next_travel_time, 2)  # Assign travel time to next attraction
        })
        
        # Update current time
        current_time += (predicted_wait_time + ride_duration + travel_time) / 60  # Convert minutes to hours
    
    # Compute overall score using a weighted approach (Pareto-inspired)
    score = compute_score(total_wait_time, total_distance, total_crowd_penalty, itinerary, list(new_visitor["preferences"]))
    
    return schedule, score

def predict_wait_time(attraction, arrival_time, waiting_df):
    """
    Estimate the wait time for an attraction based on the visitor's arrival time.
    """
    wait_times = waiting_df[(waiting_df["ENTITY_DESCRIPTION_SHORT"] == attraction)]
    wait_times = wait_times.sort_values(by="DEB_TIME")
    
    closest_time = wait_times.iloc[(wait_times["DEB_TIME_HOUR"] - arrival_time).abs().argsort()[:1]]
    return closest_time["WAIT_TIME_MAX"].values[0] if not closest_time["WAIT_TIME_MAX"].values[0] == 0 else 2  # Default to 10 min

def get_ride_duration(attraction, waiting_df):
    ride_time = waiting_df[waiting_df["ENTITY_DESCRIPTION_SHORT"] == attraction]["UP_TIME"]
    return ride_time.values[0] if not ride_time.values[0] == 0 else 5  # Default to 5 min

def get_travel_time(attraction1, attraction2, theme_park):
    """
    Get travel time between two attractions.
    """
    distance = theme_park.get_distance(attraction1, attraction2)
    travel_time = distance / 100
    return travel_time if travel_time >=1 else 1

def get_crowd_penalty(attraction, overlapping_visitors):
    """
    Compute a penalty based on the number of other visitors likely to visit this attraction.
    """
    return overlapping_visitors[
        (overlapping_visitors["ride_preference_1"] == attraction) |
        (overlapping_visitors["ride_preference_2"] == attraction) |
        (overlapping_visitors["ride_preference_3"] == attraction)
    ].shape[0]

def compute_score(wait_time, distance, crowd_penalty, itinerary, user_preferences):
    """
    Compute a weighted score balancing waiting time, distance, crowd management, and user preferences.
    """
    w_wait = -1       # Negative weight for wait time (lower is better)
    w_distance = -0.5 # Negative weight for distance (lower is better)
    w_crowd = -2      # Higher penalty for crowded attractions
    w_pref = 3        # Positive weight for visiting top preferences first

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
        w_wait * wait_time + 
        w_distance * distance + 
        w_crowd * crowd_penalty + 
        w_pref * preference_score
    )


# Streamlit app
st.title("Amusement Park Dynamic Scheduling")
st.write(
    "Welcome to Portaventura World! Select your preferences and visit duration to get an optimized schedule."
)

# User inputs
st.sidebar.header("User Preferences")
preferences = st.sidebar.multiselect(
    "Select your top 3 preferences",
    attractions,
    default=attractions[:3],
    key="preferences_multiselect",  # Unique key for the multiselect widget
)
entry_time = st.sidebar.slider(
    "Entry Time",
    10,
    19,
    10,
    key="entry_time_slider",  # Unique key for the slider widget
)
exit_time = st.sidebar.slider(
    "Exit Time",
    10,
    19,
    19,
    key="exit_time_slider",  # Unique key for the slider widget
)

st.write("Synthetic User Data", synthetic_users)

new_visitor = {
    "preferences": [preferences[0], preferences[1], preferences[2]] if len(preferences) >= 3 else preferences,
    "entry_time": entry_time,
    "exit_time": exit_time,
}

optimized_schedule = optimize_schedule(new_visitor, synthetic_users, theme_park, waiting_df)

# Show optimized schedule for the new visitor
st.subheader("Optimized Schedule for You")
st.write("Your optimized schedule based on your preferences and entry/exit times:")

for stop in optimized_schedule:
    arrival_time_formatted = f"{int(stop['arrival_time'])}h{int((stop['arrival_time'] % 1) * 60):02d}"
    departure_time_formatted = f"{int(stop['departure_time'])}h{int((stop['departure_time'] % 1) * 60):02d}"

    st.markdown(f"### ⏰ {arrival_time_formatted} - {stop['attraction']}")
    st.write(f"**- Estimated Waiting Time:** {stop['wait_time']} min")
    st.write(f"**- Ride Duration:** {stop['ride_duration']} min")
    st.write(f"**- Travel Time to Next Attraction:** {round(stop['travel_time_to_next'], 2)} min")
    st.write(f"**- Departure Time:** {departure_time_formatted}")
    st.write('---')

# Visualize Ride Activity
def visualize_activity(ride_data, theme_park):
    # Visualize the map with capacity at selected time
    fig = theme_park.visualize(ride_data)
    st.plotly_chart(fig)


# Display the interactive map
visualize_activity(ride_capacity, theme_park)

import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

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


# Generate synthetic users
def generate_synthetic_users(num_users):
    users = []
    for _ in range(num_users):
        preferences = random.sample(attractions, 3)
        entry_time = random.randint(10, 12)
        exit_time = random.randint(16, 19)
        users.append(
            {
                "preferences": preferences,
                "entry_time": entry_time,
                "exit_time": exit_time,
            }
        )
    return pd.DataFrame(users)


# Generate schedule
def generate_schedule(preferences, entry_time, exit_time, synthetic_users):
    schedule = []
    current_time = entry_time
    for preference in preferences:
        waiting_time = random.randint(10, 30)
        schedule.append(
            {
                "attraction": preference,
                "start_time": current_time,
                "end_time": current_time + 1,
                "waiting_time": waiting_time,
            }
        )
        current_time += 1
    return schedule


# Streamlit app
st.title("Amusement Park Dynamic Scheduling")
st.write(
    "Welcome to the park! Select your preferences and visit duration to get an optimized schedule."
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

# Generate synthetic data
synthetic_users = generate_synthetic_users(100)
st.write("Synthetic User Data", synthetic_users)

# Generate schedule
schedule = generate_schedule(preferences, entry_time, exit_time, synthetic_users)
st.write("Optimized Schedule", schedule)

# Activity over time
st.header("Activity Over Time")
hours = list(range(10, 20))
activity = {
    attraction: [random.randint(0, 100) for _ in hours] for attraction in attractions
}

fig, ax = plt.subplots()
for attraction in preferences:
    ax.plot(hours, activity[attraction], label=attraction)
ax.set_xlabel("Hour of the Day")
ax.set_ylabel("Number of Users")
ax.legend()
st.pyplot(fig)

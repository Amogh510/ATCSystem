import streamlit as st
st.set_page_config(layout="wide")

import threading
import time
import math
import pandas as pd
import numpy as np
import os
import pickle
import copy
import cv2 # Make sure opencv-python is installed
from streamlit_autorefresh import st_autorefresh

# Add these imports near the top
import os
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory # Optional: For agents needing memory
from dotenv import load_dotenv

# --- Configuration ---
REFRESH_INTERVAL = 1000 # 1 second
DATA_FILE = "atc_state_no_gates_v3.pkl" # New filename for this version
MAP_WIDTH_PX = 4000
MAP_HEIGHT_PX = 4000
MAP_SIZE_KM = 50
PIXELS_PER_KM = MAP_WIDTH_PX / MAP_SIZE_KM

# --- Constants ---
RUNWAY_LENGTH_KM = 1.2
RUNWAY_WIDTH_PX = 40
WAYPOINT_ARRIVAL_THRESHOLD_PX = 75 # Use the increased value
APPROACH_SPEED_KNOTS = 160.0 # Initial approach speed
DEFAULT_FLIGHT_SPEED_KNOTS = 250.0
CLIMB_OUT_DISTANCE_KM = 5.0
ALTITUDE_UPDATE_RATE_FPM = 1500.0 # Fixed descent/climb rate
TAKEOFF_DELAY_SECONDS = 10.0
LANDING_SLOWDOWN_DISTANCE_KM = 5.0
LANDING_SPEED_KNOTS = 50.0 # Final approach speed within 5km - UPDATED

# Flight Status Constants
STATUS_EN_ROUTE = "En Route"
STATUS_APPROACHING = "Approaching"
STATUS_ON_GROUND = "On Ground" # Stable status after landing
STATUS_PREPARING_TAKEOFF = "Preparing for Takeoff" # During 10s delay
STATUS_DEPARTING = "Departing" # Actively taking off
STATUS_FOLLOWING_WAYPOINTS = "Following Waypoints"

# Altitude Color Bands (BGR Format for OpenCV)
COLOR_ALT_LOW = (0, 100, 0)      # Dark Green (0-5000 ft)
COLOR_ALT_MEDIUM_LOW = (0, 165, 255) # Orange (5001-15000 ft)
COLOR_ALT_MEDIUM_HIGH = (255, 100, 0) # Blue (15001-25000 ft)
COLOR_ALT_HIGH = (0, 0, 255)      # Red (> 25000 ft)

# --- Data Initialization & Persistence ---
data_lock = threading.Lock()

def load_data():
    # (Same load_data function as before - handles missing fields)
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                data = pickle.load(f); flights = data.get('flights', {}); runways = data.get('runways', {})
                print("Loaded data from file.")
                default_runway_length_px = RUNWAY_LENGTH_KM * PIXELS_PER_KM
                for rwy_data in runways.values(): # Default checks for runways
                    rwy_data.setdefault('angle_deg', 90.0); rwy_data.setdefault('length_px', default_runway_length_px)
                    rwy_data.setdefault('width_px', RUNWAY_WIDTH_PX); rwy_data.setdefault('status', 'Available'); rwy_data.setdefault('flight_id', None)
                for flt_data in flights.values(): # Default checks for flights
                    flt_data.setdefault('waypoints', []); flt_data.setdefault('current_waypoint_index', -1)
                    flt_data.setdefault('status', STATUS_EN_ROUTE); flt_data.setdefault('target_runway', None)
                    flt_data.setdefault('altitude', 30000.0); flt_data.setdefault('speed', DEFAULT_FLIGHT_SPEED_KNOTS)
                    flt_data.setdefault('direction', 90.0); flt_data.setdefault('takeoff_clearance_time', None)
                return flights, runways
        except Exception as e: print(f"Error loading data file: {e}. Initializing fresh state.")
    else: print("No data file found. Initializing fresh state.")
    # Default initial state
    runway_length_px = RUNWAY_LENGTH_KM * PIXELS_PER_KM; flights = {}
    runways = { "RW27L": {"status": "Available", "flight_id": None, "x": 2000, "y": 3800, "angle_deg": 270.0, "length_px": runway_length_px, "width_px": RUNWAY_WIDTH_PX}, "RW09R": {"status": "Available", "flight_id": None, "x": 2000, "y": 3000, "angle_deg": 90.0, "length_px": runway_length_px, "width_px": RUNWAY_WIDTH_PX} }
    return flights, runways

def save_data(flights_data, runways_data):
    # (Same save_data as before)
    with data_lock: data_to_save = { 'flights': copy.deepcopy(flights_data), 'runways': copy.deepcopy(runways_data) }
    try:
        with open(DATA_FILE, "wb") as f: pickle.dump(data_to_save, f)
    except Exception as e: print(f"Error saving data: {e}")

# --- Helper Functions ---
def calculate_endpoint(x_center, y_center, length_px, angle_deg):
    # (Same as before)
    angle_rad = math.radians(90 - angle_deg); half_len = length_px / 2.0
    dx = half_len * math.cos(angle_rad); dy = half_len * math.sin(angle_rad)
    x1, y1 = x_center - dx, y_center + dy; x2, y2 = x_center + dx, y_center - dy
    return (int(x1), int(y1)), (int(x2), int(y2)) # p1=threshold, p2=departure

# Removed get_runway_approach_point as it's no longer used for landing waypoints

def get_runway_slowdown_point(runway_data, slowdown_dist_km=LANDING_SLOWDOWN_DISTANCE_KM):
    # (Same as before)
    threshold_pt, _ = calculate_endpoint(runway_data['x'], runway_data['y'], runway_data['length_px'], runway_data['angle_deg'])
    thresh_x, thresh_y = threshold_pt
    approach_angle_deg = (runway_data['angle_deg'] + 180) % 360; approach_angle_rad = math.radians(90 - approach_angle_deg)
    offset_px = slowdown_dist_km * PIXELS_PER_KM
    offset_dx = offset_px * math.cos(approach_angle_rad); offset_dy = offset_px * math.sin(approach_angle_rad)
    slowdown_x = thresh_x + offset_dx; slowdown_y = thresh_y - offset_dy
    return (int(slowdown_x), int(slowdown_y))

def get_runway_departure_points(runway_data, climb_out_dist_km=CLIMB_OUT_DISTANCE_KM):
    # (Same as before)
    _ , departure_threshold_pt = calculate_endpoint(runway_data['x'], runway_data['y'], runway_data['length_px'], runway_data['angle_deg'])
    dep_x, dep_y = departure_threshold_pt
    departure_angle_deg = runway_data['angle_deg']; departure_angle_rad = math.radians(90 - departure_angle_deg)
    offset_px = climb_out_dist_km * PIXELS_PER_KM
    offset_dx = offset_px * math.cos(departure_angle_rad); offset_dy = offset_px * math.sin(departure_angle_rad)
    climb_out_x = dep_x + offset_dx; climb_out_y = dep_y - offset_dy
    return departure_threshold_pt, (int(climb_out_x), int(climb_out_y))

# --- Agent Tools Definition ---
# These functions will be wrapped by LangChain tools for the agents
# They interact directly with the session state, using the data_lock

def get_all_aircraft_info():
    """Retrieves the current state of all aircraft in the simulation."""
    with data_lock:
        return copy.deepcopy(st.session_state.flights)

def get_all_runway_info():
    """Retrieves the current state of all runways."""
    with data_lock:
        return copy.deepcopy(st.session_state.runways)

def set_waypoints(flight_id: str, waypoints: list[tuple[int, int]]):
    """Sets or updates the waypoints for a specific flight.
    This cancels any current landing/takeoff procedures and sets the flight status
    to Following Waypoints.
    """
    if not isinstance(flight_id, str) or not flight_id:
        return {"error": "Invalid flight_id provided."}
    if not isinstance(waypoints, list):
         return {"error": "Waypoints must be a list."}
    valid_waypoints = []
    for wp in waypoints:
        if isinstance(wp, (list, tuple)) and len(wp) == 2:
            try:
                wp_x, wp_y = int(wp[0]), int(wp[1])
                if 0 <= wp_x < MAP_WIDTH_PX and 0 <= wp_y < MAP_HEIGHT_PX:
                    valid_waypoints.append((wp_x, wp_y))
                else:
                    return {"error": f"Waypoint ({wp_x},{wp_y}) is out of map bounds ({MAP_WIDTH_PX}x{MAP_HEIGHT_PX})."}
            except (ValueError, TypeError):
                return {"error": f"Invalid waypoint format: {wp}. Expected (x, y)."}
        else:
            return {"error": f"Invalid waypoint format: {wp}. Expected (x, y)."}

    if not valid_waypoints:
         return {"info": "No valid waypoints provided to set."} # Not necessarily an error if list was empty

    with data_lock:
        if flight_id not in st.session_state.flights:
            return {"error": f"Flight {flight_id} not found."}

        flight_data = st.session_state.flights[flight_id]
        msg_prefix = f"Waypoints set for {flight_id}."
        freed_resources = []

        # Free runway if previously assigned for landing/takeoff
        old_rwy = flight_data.get('target_runway')
        if old_rwy and old_rwy in st.session_state.runways and st.session_state.runways[old_rwy].get('flight_id') == flight_id:
            st.session_state.runways[old_rwy]['status'] = 'Available'
            st.session_state.runways[old_rwy]['flight_id'] = None
            freed_resources.append(f"Runway {old_rwy}")
            print(f"INFO (SetWaypoints): Runway {old_rwy} freed by new waypoints for {flight_id}.")

        # Update flight state
        flight_data['waypoints'] = valid_waypoints
        flight_data['current_waypoint_index'] = 0
        flight_data['status'] = STATUS_FOLLOWING_WAYPOINTS
        flight_data['speed'] = DEFAULT_FLIGHT_SPEED_KNOTS # Resume normal speed
        flight_data['target_runway'] = None
        flight_data['takeoff_clearance_time'] = None # Cancel takeoff timer

        if freed_resources:
            msg_prefix += f" ({', '.join(freed_resources)} freed)."

        print(f"INFO (SetWaypoints): {msg_prefix}")
        # Trigger save outside the tool, maybe after orchestrator runs
        # save_data(st.session_state.flights, st.session_state.runways)
        return {"success": msg_prefix}


def initiate_landing(flight_id: str, runway_id: str):
    """Directs an existing flight to land on a specified available runway."""
    if not isinstance(flight_id, str) or not flight_id:
        return {"error": "Invalid flight_id provided."}
    if not isinstance(runway_id, str) or not runway_id:
        return {"error": "Invalid runway_id provided."}

    with data_lock:
        if flight_id not in st.session_state.flights:
            return {"error": f"Flight {flight_id} not found."}
        if runway_id not in st.session_state.runways:
            return {"error": f"Runway {runway_id} not found."}
        if st.session_state.runways[runway_id].get('status') != 'Available':
            return {"error": f"Runway {runway_id} is not available (Status: {st.session_state.runways[runway_id].get('status')}, Flight: {st.session_state.runways[runway_id].get('flight_id')})."}
        if st.session_state.flights[flight_id].get('status') == STATUS_ON_GROUND:
             return {"error": f"Flight {flight_id} is already On Ground."}
        if st.session_state.flights[flight_id].get('status') == STATUS_PREPARING_TAKEOFF:
             return {"error": f"Flight {flight_id} is preparing for takeoff."}
        if st.session_state.flights[flight_id].get('status') == STATUS_DEPARTING:
             return {"error": f"Flight {flight_id} is currently departing."}


        flight_data = st.session_state.flights[flight_id]
        new_runway_data = st.session_state.runways[runway_id]
        freed_resources=[]

        # Free previous runway if assigned
        old_rwy = flight_data.get('target_runway')
        if old_rwy and old_rwy != runway_id and old_rwy in st.session_state.runways and st.session_state.runways[old_rwy].get('flight_id') == flight_id:
            st.session_state.runways[old_rwy]['status'] = 'Available'
            st.session_state.runways[old_rwy]['flight_id'] = None
            freed_resources.append(f"Runway {old_rwy}")
            print(f"INFO (InitiateLanding): Runway {old_rwy} freed by assigning {flight_id} to {runway_id}.")

        # Calculate Landing Waypoints (Slowdown + Threshold ONLY)
        slowdown_pt = get_runway_slowdown_point(new_runway_data)
        thresh_pt, _ = calculate_endpoint(new_runway_data['x'], new_runway_data['y'], new_runway_data['length_px'], new_runway_data['angle_deg'])

        # Update flight state
        flight_data['waypoints'] = [slowdown_pt, thresh_pt]
        flight_data['current_waypoint_index'] = 0
        flight_data['status'] = STATUS_APPROACHING
        flight_data['speed'] = APPROACH_SPEED_KNOTS # Start approach fast
        flight_data['target_runway'] = runway_id
        flight_data['takeoff_clearance_time'] = None # Clear any takeoff timer

        # Occupy runway
        new_runway_data['status'] = 'Occupied'
        new_runway_data['flight_id'] = flight_id

        msg = f"Directing {flight_id} to land on {runway_id}."
        if freed_resources:
            msg += f" ({', '.join(freed_resources)} freed)."

        print(f"INFO (InitiateLanding): {msg}")
        # Trigger save outside the tool
        # save_data(st.session_state.flights, st.session_state.runways)
        return {"success": msg}

def initiate_takeoff(flight_id: str, runway_id: str):
    """Commands a flight currently 'On Ground' to take off from a specified available runway."""
    if not isinstance(flight_id, str) or not flight_id:
        return {"error": "Invalid flight_id provided."}
    if not isinstance(runway_id, str) or not runway_id:
        return {"error": "Invalid runway_id provided."}

    with data_lock:
        if flight_id not in st.session_state.flights:
            return {"error": f"Flight {flight_id} not found."}
        if runway_id not in st.session_state.runways:
            return {"error": f"Runway {runway_id} not found."}
        if st.session_state.flights[flight_id].get('status') != STATUS_ON_GROUND:
            return {"error": f"Flight {flight_id} is not 'On Ground' (Status: {st.session_state.flights[flight_id].get('status')})."}
        if st.session_state.runways[runway_id].get('status') != 'Available':
            return {"error": f"Runway {runway_id} is not available (Status: {st.session_state.runways[runway_id].get('status')}, Flight: {st.session_state.runways[runway_id].get('flight_id')})."}

        flight_data = st.session_state.flights[flight_id]
        runway_data = st.session_state.runways[runway_id]

        # Occupy runway
        runway_data['status'] = 'Occupied'
        runway_data['flight_id'] = flight_id

        # Get departure threshold point and position the aircraft there
        dep_thresh_pt, _ = get_runway_departure_points(runway_data) # We only need the threshold point for positioning
        dep_x, dep_y = dep_thresh_pt

        # Update flight state for takeoff prep
        flight_data['status'] = STATUS_PREPARING_TAKEOFF
        flight_data['speed'] = 0.0
        flight_data['altitude'] = 0.0
        flight_data['x'] = float(dep_x) # Position at departure end
        flight_data['y'] = float(dep_y)
        flight_data['target_runway'] = runway_id
        flight_data['takeoff_clearance_time'] = time.time() # Start the delay timer
        flight_data['waypoints'] = [] # Clear any previous waypoints
        flight_data['current_waypoint_index'] = -1
        # Direction will be set when STATUS_DEPARTING begins in update_flight_positions

        msg = f"Flight {flight_id} preparing for takeoff from {runway_id} ({TAKEOFF_DELAY_SECONDS}s delay)."
        print(f"INFO (InitiateTakeoff): {msg}")
        # Trigger save outside the tool
        # save_data(st.session_state.flights, st.session_state.runways)
        return {"success": msg}

# --- End of Agent Tools Definition ---


# --- Agent Setup ---

@st.cache_resource # Cache the results of this function
def initialize_agents_and_llm():
    """Initializes the LLM, tools, and agents once and caches them."""
    print("--- Running Agent Initialization (should only happen once) ---")
    initialized_resources = {
        "llm": None,
        "tools": [],
        "scheduler_agent": None,
        "conflict_agent": None
    }

    # 1. Initialize LLM
    try:
        # load_dotenv() should have been called outside this function already
        llm = ChatGroq(temperature=0.1, model_name="llama3-70b-8192")
        print("Groq LLM Initialized.")
        initialized_resources["llm"] = llm
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {e}. Is GROQ_API_KEY set?")
        print(f"ERROR: Failed to initialize Groq LLM: {e}")
        # Return partially initialized resources so app can maybe still run parts
        return initialized_resources

    # 2. Define LangChain Tools
    current_tools = [
         Tool(
            name="GetAllAircraftInfo",
            func=lambda _: get_all_aircraft_info(),
            description="Retrieves the current state (position, altitude, speed, status, waypoints, target runway) of all aircraft in the simulation.",
        ),
        Tool(
            name="GetAllRunwayInfo",
            func=lambda _: get_all_runway_info(),
            description="Retrieves the current state (status, occupying flight ID) of all runways.",
        ),
        Tool(
            name="SetWaypoints",
            func=lambda args: set_waypoints(flight_id=args.get('flight_id'), waypoints=args.get('waypoints')),
            description="Sets or updates the waypoints for a specific flight (flight_id: str, waypoints: list[tuple[int, int]]). Cancels any landing/takeoff and sets status to Following Waypoints.",
        ),
        Tool(
            name="InitiateLanding",
            func=lambda args: initiate_landing(flight_id=args.get('flight_id'), runway_id=args.get('runway_id')),
            description="Directs an existing non-departing/non-ground flight (flight_id: str) to land on a specified AVAILABLE runway (runway_id: str). Sets status to Approaching and assigns landing waypoints.",
        ),
        Tool(
            name="InitiateTakeoff",
            func=lambda args: initiate_takeoff(flight_id=args.get('flight_id'), runway_id=args.get('runway_id')),
            description="Commands a flight currently 'On Ground' (flight_id: str) to take off from a specified AVAILABLE runway (runway_id: str). Sets status to Preparing for Takeoff.",
        ),
    ]
    print(f"LangChain Tools Defined: {[tool.name for tool in current_tools]}")
    initialized_resources["tools"] = current_tools

    # 3. Initialize Agents
    # -- Scheduler Agent --
    # NOTE: Keep the original multiline string prompt definitions here
    scheduler_prompt_template = """You are an expert Air Traffic Control Scheduler. Your goal is to manage landings and takeoffs efficiently and safely using the available runways.

    Available Tools:
    {tools}

    Current Airspace State:
    {input}

    Thought Process:
    1. Analyze the current aircraft and runway states provided in the input.
    2. Identify available runways.
    3. Identify flights requesting or suitable for landing (Status: En Route, Following Waypoints near the airport - let's assume any non-approaching/departing/ground flight is potentially eligible).
    4. Identify flights ready for takeoff (Status: On Ground).
    5. Prioritize landing requests if runways are available. If multiple flights want to land, pick one (e.g., the first one listed for simplicity).
    6. If no landings are pending/possible but takeoffs are ready and a runway is free, schedule a takeoff.
    7. If a flight needs to land but no runway is available, consider putting it in a holding pattern by assigning waypoints. A simple holding pattern can be 3 waypoints forming a small loop near its current position (e.g., current_pos -> point A -> point B -> current_pos). Calculate suitable coordinates for point A and point B (e.g., 5km North and 5km East of current_pos).
    8. You can only issue ONE landing or ONE takeoff command per cycle using InitiateLanding or InitiateTakeoff.
    9. ONLY use the provided tools. Do not make up information. Provide the required arguments exactly (flight_id as string, runway_id as string, waypoints as list of tuples).
    10. Respond with your action or state that no action is needed. Use the tools to perform actions.

    Begin!

    {agent_scratchpad}
    """
    SCHEDULER_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", scheduler_prompt_template),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    try:
        scheduler_agent = initialize_agent(
            tools=current_tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prompt=SCHEDULER_PROMPT, verbose=True, handle_parsing_errors=True,
        )
        print("Scheduler Agent Initialized.")
        initialized_resources["scheduler_agent"] = scheduler_agent
    except Exception as e:
        print(f"Error initializing Scheduler Agent: {e}")
        st.error(f"Failed to initialize Scheduler Agent: {e}")

    # -- Conflict Resolution Agent --
    # NOTE: Keep the original multiline string prompt definition here
    conflict_prompt_template = """You are an expert Air Traffic Control Conflict Resolution specialist. Your goal is to prevent aircraft from getting too close to each other (loss of separation). Assume minimum safe separation is 150 pixels horizontally OR 1000 feet vertically.

    Available Tools:
    {tools}

    Current Airspace State:
    {input}

    Thought Process (Chain-of-Thought):
    1. Analyze the current state of all active flights (not On Ground or Preparing Takeoff).
    2. For each pair of active flights, calculate the current horizontal distance (in pixels) and vertical distance (in feet).
    3. Predict potential conflicts: Check if any pair is currently closer than 150px horizontally AND closer than 1000ft vertically. (Focus on current separation for now, prediction is complex).
    4. If a potential conflict is detected between Flight A and Flight B:
        a. Identify the flights involved (flight_id).
        b. Note their current positions, altitudes, and statuses.
        c. Decide on a resolution strategy. The simplest is to vector (turn) one of the aircraft slightly using SetWaypoints. Choose the aircraft that is NOT currently landing or taking off, if possible.
        d. Calculate a safe vector: Assign a single waypoint slightly off the current aircraft's track (e.g., 5km ahead and 30 degrees to the right of its current heading). Calculate the (x, y) coordinates for this waypoint.
        e. Use the SetWaypoints tool to issue the vector to the chosen aircraft. Provide flight_id and the calculated waypoint as a list containing one tuple: `[(x, y)]`.
    5. Handle only ONE conflict per cycle. If multiple conflicts exist, resolve the most critical (closest pair) first and stop.
    6. If no conflicts are detected, state that clearly.
    7. ONLY use the provided tools. Do not make up information. Ensure arguments are correct.

    Begin!

    {agent_scratchpad}
    """
    CONFLICT_PROMPT = ChatPromptTemplate.from_messages(
         [
            ("system", conflict_prompt_template),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    try:
        conflict_agent = initialize_agent(
            tools=current_tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prompt=CONFLICT_PROMPT, verbose=True, handle_parsing_errors=True,
        )
        print("Conflict Resolution Agent Initialized.")
        initialized_resources["conflict_agent"] = conflict_agent
    except Exception as e:
        print(f"Error initializing Conflict Agent: {e}")
        st.error(f"Failed to initialize Conflict Agent: {e}")

    # Communication agent setup remains minimal as it uses direct LLM call
    print("Communication Agent will use direct LLM call (no specific init needed here).")

    print("--- Agent Initialization Complete ---")
    return initialized_resources

# Call the cached function ONCE
agent_resources = initialize_agents_and_llm()


# --- Orchestrator Function (now split) ---

def _agent_cycle_thread_target(agent_resources):
    """Target function for the background agent execution thread."""
    print("\n--- Background Agent Cycle Started ---")
    # Access resources from the passed dictionary
    llm = agent_resources.get("llm")
    scheduler_agent = agent_resources.get("scheduler_agent")
    conflict_agent = agent_resources.get("conflict_agent")
    # Check if agents/llm are None before trying to use them
    if not llm or not scheduler_agent or not conflict_agent:
         print("AgentThread: ERROR - Required agent resources not initialized. Exiting cycle.")
         comm_message = "ERROR: Agents not initialized."
         with data_lock:
             st.session_state.agent_status_message = comm_message
             st.session_state.agent_cycle_running = False
         return

    action_details = {}
    action_taken_this_cycle = False
    comm_message = "Agent cycle running..." # Default message

    try: # Wrap the entire cycle for broad error catching
        # 1. Get Current State
        print("AgentThread: Getting current state...")
        current_aircraft_info = get_all_aircraft_info()
        current_runway_info = get_all_runway_info()

        if not current_aircraft_info:
            print("AgentThread: No aircraft detected. Exiting cycle.")
            comm_message = "No aircraft; agents idle."
            with data_lock: st.session_state.agent_status_message = comm_message
            return # Exit thread

        current_state_summary = f"Current Time: {time.time():.0f}\nAircraft: {current_aircraft_info}\nRunways: {current_runway_info}"
        print(f"AgentThread: Current State Snapshot:\n{current_state_summary}")

        # --- 2. Run Scheduler Agent ---
        print("AgentThread: Running SCHEDULER Agent...")
        # Use the scheduler_agent variable obtained from agent_resources
        scheduler_input_dict = {"input": current_state_summary}
        try:
            scheduler_response = scheduler_agent.invoke(scheduler_input_dict)
            scheduler_output = scheduler_response.get('output', '')
            print(f"Scheduler Agent Raw Output: {scheduler_output}")
            # --- Parsing logic (same as before) ---
            if "InitiateLanding successfully" in scheduler_output or "Successfully directed" in scheduler_output:
                parts = scheduler_output.split()
                try:
                    flight_id = parts[parts.index("landing") - 1]
                    runway_id = parts[parts.index("on") + 1].replace('.','')
                    action_details[flight_id] = f"cleared approach runway {runway_id}"
                    action_taken_this_cycle = True
                except (ValueError, IndexError): action_details["Scheduler"] = "Initiated landing (details unclear)"; action_taken_this_cycle = True
            elif "InitiateTakeoff successfully" in scheduler_output or "preparing for takeoff" in scheduler_output:
                parts = scheduler_output.split()
                try:
                    flight_id = parts[parts.index("takeoff") - 1]
                    runway_id = parts[parts.index("from") + 1].replace('.','')
                    action_details[flight_id] = f"cleared for takeoff runway {runway_id}"
                    action_taken_this_cycle = True
                except (ValueError, IndexError): action_details["Scheduler"] = "Initiated takeoff (details unclear)"; action_taken_this_cycle = True
            elif "SetWaypoints successfully" in scheduler_output or "Waypoints set for" in scheduler_output:
                parts = scheduler_output.split()
                try:
                     flight_id = parts[parts.index("for") + 1].replace('.','')
                     action_details[flight_id] = "assigned new waypoints (holding/vectoring)"
                     action_taken_this_cycle = True
                except (ValueError, IndexError): action_details["Scheduler"] = "Set waypoints (details unclear)"; action_taken_this_cycle = True
            else: print("AgentThread: Scheduler agent did not report taking a major action.")
        except Exception as e_sched: print(f"ERROR running Scheduler Agent: {e_sched}")

        # --- 3. Run Conflict Resolution Agent ---
        print("\nAgentThread: Getting updated state for Conflict Agent...")
        current_aircraft_info = get_all_aircraft_info() # Re-get potentially updated state
        current_runway_info = get_all_runway_info()
        active_flights_count = sum(1 for data in current_aircraft_info.values() if data.get('status') not in [STATUS_ON_GROUND, STATUS_PREPARING_TAKEOFF])

        if active_flights_count >= 2:
            print("AgentThread: Running CONFLICT RESOLUTION Agent...")
            # Use the conflict_agent variable obtained from agent_resources
            conflict_state_summary = f"Current Time: {time.time():.0f}\nAircraft: {current_aircraft_info}\nRunways: {current_runway_info}"
            conflict_input_dict = {"input": conflict_state_summary}
            try:
                conflict_response = conflict_agent.invoke(conflict_input_dict)
                conflict_output = conflict_response.get('output', '')
                print(f"Conflict Agent Raw Output: {conflict_output}")
                 # --- Parsing logic (same as before) ---
                if "SetWaypoints successfully" in conflict_output or "Waypoints set for" in conflict_output:
                     parts = conflict_output.split()
                     try:
                         flight_id = parts[parts.index("for") + 1].replace('.','')
                         if flight_id not in action_details:
                             action_details[flight_id] = "vector for traffic separation"
                             action_taken_this_cycle = True
                     except (ValueError, IndexError):
                          if "Conflict" not in action_details: action_details["Conflict"] = "Set waypoints for conflict (details unclear)"; action_taken_this_cycle = True
                elif "No conflicts detected" in conflict_output: print("AgentThread: Conflict agent reported no conflicts.")
                else: print("AgentThread: Conflict agent finished, but no clear action or no-conflict statement parsed.")
            except Exception as e_conf: print(f"ERROR running Conflict Resolution Agent: {e_conf}")
        elif active_flights_count < 2: print("AgentThread: Skipping conflict resolution (fewer than 2 active aircraft).")

        # --- 4. Generate Communication ---
        comm_message = "No actions taken by agents." # Default if no actions
        if action_details:
            print("\nAgentThread: Running COMMUNICATION Generation...")
            communication_prompt_text = "You are an ATC communication specialist. Convert the following structured ATC actions into concise, realistic radio calls. Use standard phraseology. Address each flight individually.\n\nActions Taken:\n"
            for flight_id, desc in action_details.items(): communication_prompt_text += f"- {flight_id}: {desc}\n"
            communication_prompt_text += "\nGenerated Radio Calls:"
            try:
                # Use the llm variable obtained from agent_resources
                communication_response = llm.invoke(communication_prompt_text)
                if hasattr(communication_response, 'content'): final_comm = communication_response.content
                else: final_comm = str(communication_response)
                print(f"Communication LLM Raw Output: {final_comm}")
                comm_message = f"Agent Communications:\n```\n{final_comm}\n```"
            except Exception as e_comm:
                print(f"ERROR running Communication LLM: {e_comm}")
                fallback_comm = ["ATC Actions (Comm Gen Error):"] + [f"- {fid}: {desc}" for fid, desc in action_details.items()]
                comm_message = f"Agent Communications (Fallback):\n```\n" + "\n".join(fallback_comm) + "\n```"
        else:
            print("AgentThread: No actions taken by agents, skipping communication generation.")

        # --- 5. Save State ---
        if action_taken_this_cycle:
            print("\nAgentThread: Saving final state...")
            save_data(st.session_state.flights, st.session_state.runways)
            print("AgentThread: State saved.")
        else:
            print("\nAgentThread: No state changes detected by agents, skipping save.")

    except Exception as e_cycle:
         print(f"ERROR in background agent cycle: {e_cycle}")
         comm_message = f"Agent cycle failed: {e_cycle}"

    # Update session state for UI feedback *after* cycle completes or fails
    with data_lock:
        st.session_state.agent_status_message = comm_message
        st.session_state.agent_cycle_running = False

    print("--- Background Agent Cycle Finished ---")

def run_atc_orchestrator(agent_resources):
    """Starts the agent execution cycle in a background thread."""
    if st.session_state.get('agent_cycle_running', False):
         st.warning("Agent cycle is already running in the background.")
         return

    # Check if resources were initialized before starting thread
    if not agent_resources or not agent_resources.get("llm"):
         st.error("Cannot start agent cycle: LLM or essential resources failed to initialize.")
         return

    print("UI Thread: Received request to start agent cycle.")
    st.session_state.agent_cycle_running = True
    st.session_state.agent_status_message = "Agent cycle starting..."

    # Pass agent_resources to the target function
    agent_thread = threading.Thread(target=_agent_cycle_thread_target, args=(agent_resources,), daemon=True)
    agent_thread.start()

    print("UI Thread: Background agent thread started. UI remains responsive.")
    st.info("ATC Agent cycle running in background... Refresh page periodically or check console logs. Status message will update below.")
    # No st.rerun() here, let autorefresh handle UI updates or user interaction


# --- End of Agent Setup ---


# --- Background Flight Position Updater ---
def update_flight_positions(flights_dict, runways_dict):
    LANDING_SLOWDOWN_DISTANCE_PX = LANDING_SLOWDOWN_DISTANCE_KM * PIXELS_PER_KM

    while True:
        start_time = time.time()
        updates_to_apply = {}
        flights_to_remove = []
        alt_change_per_sec = ALTITUDE_UPDATE_RATE_FPM / 60.0
        time_delta_sec = REFRESH_INTERVAL / 1000.0

        with data_lock:
            current_flights_items = list(flights_dict.items())

        for flight_id, data in current_flights_items:
            status = data.get('status', 'Unknown')

            # --- Handle Preparing for Takeoff Timer ---
            if status == STATUS_PREPARING_TAKEOFF:
                takeoff_time = data.get('takeoff_clearance_time')
                if takeoff_time and time.time() >= takeoff_time + TAKEOFF_DELAY_SECONDS:
                    # Start Takeoff Roll (Logic unchanged)
                    print(f"DEBUG {flight_id}: Takeoff delay over. Starting departure roll.")
                    prep_updates = {'status': STATUS_DEPARTING, 'speed': DEFAULT_FLIGHT_SPEED_KNOTS, 'takeoff_clearance_time': None}
                    target_runway_id = data.get('target_runway')
                    if target_runway_id and target_runway_id in runways_dict:
                        runway_data = runways_dict[target_runway_id]
                        _ , climb_out_pt = get_runway_departure_points(runway_data)
                        prep_updates['waypoints'] = [climb_out_pt]; prep_updates['current_waypoint_index'] = 0
                        dep_thresh_x, dep_thresh_y = data['x'], data['y']
                        climb_dx = climb_out_pt[0]-dep_thresh_x; climb_dy = climb_out_pt[1]-dep_thresh_y
                        if climb_dx != 0 or climb_dy != 0:
                             angle_rad = math.atan2(-climb_dy, climb_dx); angle_deg = math.degrees(angle_rad)
                             prep_updates['direction'] = (90 - angle_deg + 360) % 360
                    else: prep_updates['status'] = STATUS_EN_ROUTE
                    updates_to_apply[flight_id] = prep_updates
                continue # Skip other processing

            # --- Skip if On Ground ---
            if status == STATUS_ON_GROUND:
                continue

            # --- Process Moving/Active Flights ---
            if 'x' not in data or 'y' not in data: continue

            current_x, current_y = data['x'], data['y']
            speed_knots = data.get('speed', 0.0)
            current_direction = data.get('direction', 0.0)
            waypoints = data.get('waypoints', [])
            wp_index = data.get('current_waypoint_index', -1)
            current_alt = data.get('altitude', 0.0)
            target_runway_id = data.get('target_runway')

            calculated_direction = current_direction
            final_updates = {}
            is_approaching = (status == STATUS_APPROACHING)

            # --- Landing Speed Adjustment ---
            if is_approaching and target_runway_id and target_runway_id in runways_dict:
                runway_data = runways_dict[target_runway_id]
                threshold_pt, _ = calculate_endpoint(runway_data['x'], runway_data['y'], runway_data['length_px'], runway_data['angle_deg'])
                dist_to_thresh = math.sqrt((threshold_pt[0] - current_x)**2 + (threshold_pt[1] - current_y)**2)

                # If within slowdown distance AND faster than target landing speed, slow down
                if dist_to_thresh <= LANDING_SLOWDOWN_DISTANCE_PX:
                    if speed_knots > LANDING_SPEED_KNOTS:
                         print(f"DEBUG {flight_id}: Within {LANDING_SLOWDOWN_DISTANCE_KM}km. Reducing speed to {LANDING_SPEED_KNOTS} kts.")
                         final_updates['speed'] = LANDING_SPEED_KNOTS
                         speed_knots = LANDING_SPEED_KNOTS # Use updated speed for movement
                # No need for an else here; speed remains APPROACH_SPEED_KNOTS until this point

            # --- Waypoint Navigation ---
            if wp_index >= 0 and wp_index < len(waypoints):
                target_x, target_y = waypoints[wp_index]
                dx = target_x - current_x; dy = target_y - current_y
                distance_to_target = math.sqrt(dx**2 + dy**2)

                if distance_to_target <= WAYPOINT_ARRIVAL_THRESHOLD_PX:
                    # --- LANDING THRESHOLD ARRIVAL ---
                    # The landing threshold is now always the LAST waypoint in the landing sequence
                    if is_approaching and wp_index == len(waypoints) - 1:
                        print(f"DEBUG {flight_id}: Reached landing threshold. Setting status to On Ground.")
                        final_updates['status'] = STATUS_ON_GROUND
                        final_updates['speed'] = 0.0
                        final_updates['altitude'] = 0.0
                        final_updates['waypoints'] = []
                        final_updates['current_waypoint_index'] = -1
                        final_updates['x'] = float(target_x) # Snap position exactly
                        final_updates['y'] = float(target_y)
                        # Release runway
                        if target_runway_id in runways_dict:
                            with data_lock: # Lock needed for write
                                if runways_dict[target_runway_id].get('flight_id') == flight_id:
                                    runways_dict[target_runway_id]['status'] = 'Available'
                                    runways_dict[target_runway_id]['flight_id'] = None
                                    print(f"DEBUG {flight_id}: Runway {target_runway_id} released on landing.")
                        final_updates['target_runway'] = None
                        speed_knots = 0 # Prevent movement

                    # --- REGULAR WAYPOINT ARRIVAL ---
                    else:
                        final_updates['current_waypoint_index'] = wp_index + 1
                        wp_index += 1

                        if wp_index >= len(waypoints): # Reached final regular waypoint
                            final_updates['waypoints'] = []
                            final_updates['current_waypoint_index'] = -1
                            if status == STATUS_DEPARTING: # Reached climb-out point
                                final_updates['status'] = STATUS_EN_ROUTE
                                final_updates['speed'] = DEFAULT_FLIGHT_SPEED_KNOTS
                                if target_runway_id in runways_dict: # Release departure runway
                                    with data_lock:
                                        if runways_dict[target_runway_id].get('flight_id') == flight_id:
                                            runways_dict[target_runway_id]['status'] = 'Available'
                                            runways_dict[target_runway_id]['flight_id'] = None
                                            print(f"Runway {target_runway_id} released by departing {flight_id}.")
                                    final_updates['target_runway'] = None
                            elif status == STATUS_FOLLOWING_WAYPOINTS:
                                final_updates['status'] = STATUS_EN_ROUTE
                        else: # Proceeding to next waypoint, update direction
                            next_target_x, next_target_y = waypoints[wp_index]
                            next_dx = next_target_x - current_x; next_dy = next_target_y - current_y
                            if next_dx != 0 or next_dy != 0:
                                angle_rad = math.atan2(-next_dy, next_dx); angle_deg = math.degrees(angle_rad)
                                calculated_direction = (90 - angle_deg + 360) % 360
                                final_updates['direction'] = calculated_direction
                else: # Still heading towards current waypoint, update direction
                    if dx != 0 or dy != 0:
                        angle_rad = math.atan2(-dy, dx); angle_deg = math.degrees(angle_rad)
                        calculated_direction = (90 - angle_deg + 360) % 360
                        final_updates['direction'] = calculated_direction

            # --- Position Update Calculation ---
            new_x, new_y = current_x, current_y
            if speed_knots > 0:
                speed_km_s = (speed_knots * 1.852) / 3600
                speed_px_s = speed_km_s * PIXELS_PER_KM * time_delta_sec
                move_angle_rad = math.radians(90 - calculated_direction)
                delta_x = speed_px_s * math.cos(move_angle_rad); delta_y = speed_px_s * math.sin(move_angle_rad)
                new_x = current_x + delta_x
                new_y = current_y - delta_y # Screen coordinates

            # --- MAP EXIT CHECK ---
            if new_x < 0 or new_x >= MAP_WIDTH_PX or new_y < 0 or new_y >= MAP_HEIGHT_PX:
                print(f"INFO: Flight {flight_id} exited map boundaries. Removing.")
                flights_to_remove.append(flight_id)
                if target_runway_id and target_runway_id in runways_dict: # Release runway if occupied
                    with data_lock:
                        if runways_dict[target_runway_id].get('flight_id') == flight_id:
                            runways_dict[target_runway_id]['status'] = 'Available'; runways_dict[target_runway_id]['flight_id'] = None
                            print(f"INFO: Runway {target_runway_id} released by exiting flight {flight_id}.")
                continue # Skip other updates

            # --- Apply Position Update ---
            # Use setdefault to avoid overwriting speed/status updates made earlier in the loop
            if abs(new_x - current_x) > 0.1: final_updates.setdefault('x', max(0.0, min(new_x, MAP_WIDTH_PX - 1.0)))
            if abs(new_y - current_y) > 0.1: final_updates.setdefault('y', max(0.0, min(new_y, MAP_HEIGHT_PX - 1.0)))

            # --- Altitude Update (using fixed rate) ---
            if status == STATUS_DEPARTING:
                 final_updates['altitude'] = min(35000.0, current_alt + alt_change_per_sec * time_delta_sec)
            elif status == STATUS_APPROACHING:
                 final_updates['altitude'] = max(0.0, current_alt - alt_change_per_sec * time_delta_sec) # Starts descending immediately

            # Store updates for this flight
            if final_updates:
                if flight_id in updates_to_apply: updates_to_apply[flight_id].update(final_updates)
                else: updates_to_apply[flight_id] = final_updates

        # --- Apply all updates and removals ---
        if updates_to_apply or flights_to_remove:
            with data_lock:
                for flight_id, updates in updates_to_apply.items():
                    if flight_id in flights_dict and flight_id not in flights_to_remove:
                         flights_dict[flight_id].update(updates)
                state_changed = False
                for flight_id in flights_to_remove:
                    if flight_id in flights_dict: del flights_dict[flight_id]; state_changed = True
            if state_changed: save_data(st.session_state.flights, st.session_state.runways)

        # Sleep
        elapsed_time = time.time() - start_time
        sleep_time = max(0, time_delta_sec - elapsed_time)
        time.sleep(sleep_time)


# --- Initialize Data and Start Background Thread ---
if 'app_state_initialized' not in st.session_state:
    st.session_state.flights, st.session_state.runways = load_data()
    st.session_state.app_state_initialized = True; print("App state initialized.")
if 'update_thread_started' not in st.session_state:
    threading.Thread(target=update_flight_positions, args=(st.session_state.flights, st.session_state.runways), daemon=True).start()
    st.session_state.update_thread_started = True; print("Update thread started.")

st_autorefresh(interval=REFRESH_INTERVAL, key="airspace_refresh")

# --- Streamlit UI ---
st.title("ATC Flight Tracker")
col1, col2 = st.columns([3, 1])

with col2: # Controls Column
    st.header("Controls")
    # --- Add Flight Form --- (Unchanged)
    with st.form("flight_form"):
        st.subheader("Add New Flight")
        flight_id = st.text_input("Flight ID").strip().upper()
        speed = st.number_input("Initial Speed (knots)", min_value=0.0, value=DEFAULT_FLIGHT_SPEED_KNOTS, step=10.0)
        direction = st.number_input("Direction (deg)", min_value=0.0, max_value=359.9, value=90.0, step=5.0)
        x = st.slider("Initial X", 0.0, float(MAP_WIDTH_PX - 1), float(MAP_WIDTH_PX / 2))
        y = st.slider("Initial Y", 0.0, float(MAP_HEIGHT_PX - 1), float(MAP_HEIGHT_PX / 2))
        altitude = st.number_input("Altitude (ft)", min_value=0, value=30000, step=500)
        status = st.selectbox("Initial Status", [STATUS_EN_ROUTE, STATUS_DEPARTING])
        submitted = st.form_submit_button("Add Flight")
        if submitted:
            print("DEBUG: Add Flight button clicked.")
            if not flight_id:
                st.error("Flight ID required.")
                print("DEBUG: Flight ID missing.")
            elif flight_id in st.session_state.get('flights', {}):
                st.error(f"Flight {flight_id} exists.")
                print(f"DEBUG: Flight {flight_id} already exists.")
            else:
                print(f"DEBUG: Preparing new flight data for {flight_id}.")
                new_flight_data = { 'speed': float(speed), 'direction': float(direction), 'x': float(x), 'y': float(y), 'altitude': float(altitude), 'status': status, 'waypoints': [], 'current_waypoint_index': -1, 'target_runway': None, 'takeoff_clearance_time': None }
                try:
                    with data_lock:
                        print("DEBUG: Acquired data_lock for adding flight.")
                        if 'flights' not in st.session_state:
                             print("DEBUG: Initializing st.session_state.flights dictionary.")
                             st.session_state.flights = {}
                        st.session_state.flights[flight_id] = new_flight_data
                        print(f"DEBUG: Added {flight_id} to st.session_state.flights.")
                    print("DEBUG: Released data_lock.")

                    print("DEBUG: Calling save_data...")
                    save_data(st.session_state.flights, st.session_state.runways)
                    print("DEBUG: save_data finished.")

                    st.success(f"Flight {flight_id} added.")
                    print("DEBUG: Calling st.rerun()...")
                    st.rerun()
                except Exception as e_add:
                     print(f"ERROR during flight add/save for {flight_id}: {e_add}")
                     st.error(f"Error adding flight {flight_id}: {e_add}")

    # --- Flight Actions Section ---
    st.header("Flight Actions")
    with data_lock: # Get copies for UI population
        flights_copy_actions = copy.deepcopy(st.session_state.flights)
        runways_copy_actions = copy.deepcopy(st.session_state.runways)
    active_flight_ids = sorted(list(flights_copy_actions.keys()))
    selected_flight_id_action = st.selectbox("Select Flight for Action", options=[""] + active_flight_ids, key="action_select")

    if selected_flight_id_action:
        # Get current data for selected flight
        current_flight_data = flights_copy_actions.get(selected_flight_id_action, {})
        current_status = current_flight_data.get('status', 'Unknown')
        current_speed = current_flight_data.get('speed', 0.0)
        current_altitude = current_flight_data.get('altitude', 0.0)

        # --- Direct Speed/Altitude Control --- (Unchanged)
        st.subheader(f"Update {selected_flight_id_action} Params"); col_spd, col_alt = st.columns(2)
        with col_spd: new_speed = st.number_input("Set Speed (kts)", min_value=0.0, value=current_speed, step=10.0, key=f"spd_{selected_flight_id_action}")
        with col_alt: new_altitude = st.number_input("Set Altitude (ft)", min_value=0.0, value=current_altitude, step=500.0, key=f"alt_{selected_flight_id_action}")
        if st.button("Update Speed/Altitude", key=f"update_params_{selected_flight_id_action}"):
            with data_lock:
                if selected_flight_id_action in st.session_state.flights:
                    st.session_state.flights[selected_flight_id_action]['speed'] = new_speed; st.session_state.flights[selected_flight_id_action]['altitude'] = new_altitude
                    st.success(f"Updated {selected_flight_id_action}."); save_data(st.session_state.flights, st.session_state.runways)
                else: st.error("Flight not found.")
            st.rerun()
        st.divider()

        # --- Waypoint Input --- (Unchanged logic, cancels other commands)
        st.subheader(f"Waypoints for {selected_flight_id_action}")
        current_waypoints = current_flight_data.get('waypoints', []); current_waypoints_str = "\n".join([f"{int(wp[0])},{int(wp[1])}" for wp in current_waypoints])
        waypoints_str = st.text_area("Enter Waypoints (x,y per line)", value=current_waypoints_str, height=100, key=f"wp_area_{selected_flight_id_action}")
        if st.button("Set Waypoints", key=f"wp_btn_{selected_flight_id_action}"):
            waypoints = []; valid_waypoints_found = False
            try: # (Parsing loop...)
                lines = waypoints_str.strip().split('\n');
                for line in lines:
                    if line.strip(): parts = line.split(',');
                    if len(parts) == 2:
                        wp_x, wp_y = int(parts[0].strip()), int(parts[1].strip())
                        if 0 <= wp_x < MAP_WIDTH_PX and 0 <= wp_y < MAP_HEIGHT_PX: waypoints.append((wp_x, wp_y)); valid_waypoints_found = True
                        else: st.warning(f"Wpt ({wp_x},{wp_y}) out of bounds.")
                    else: st.warning(f"Invalid wpt format: '{line}'.")
                if valid_waypoints_found:
                    with data_lock:
                        if selected_flight_id_action in st.session_state.flights:
                            flight_data = st.session_state.flights[selected_flight_id_action]; msg_prefix = f"Waypoints set for {selected_flight_id_action}."; freed_resources = []
                            old_rwy = flight_data.get('target_runway') # Free runway
                            if old_rwy and old_rwy in st.session_state.runways and st.session_state.runways[old_rwy].get('flight_id') == selected_flight_id_action:
                                st.session_state.runways[old_rwy]['status'] = 'Available'; st.session_state.runways[old_rwy]['flight_id'] = None; freed_resources.append(f"Runway {old_rwy}")
                            flight_data['waypoints'] = waypoints; flight_data['current_waypoint_index'] = 0; flight_data['status'] = STATUS_FOLLOWING_WAYPOINTS
                            flight_data['speed'] = DEFAULT_FLIGHT_SPEED_KNOTS; flight_data['target_runway'] = None; flight_data['takeoff_clearance_time'] = None
                            if freed_resources: msg_prefix += f" ({', '.join(freed_resources)} freed)."
                            st.success(msg_prefix); save_data(st.session_state.flights, st.session_state.runways)
                        else: st.error("Flight not found.")
                    st.rerun()
                elif waypoints_str.strip() and not valid_waypoints_found: st.error("No valid waypoints parsed.")
                else: st.info("No waypoints entered.")
            except Exception as e: st.error(f"Error setting waypoints: {e}")
        st.divider()

        # --- Landing Command --- (Modified to set only 2 waypoints)
        st.subheader(f"Direct {selected_flight_id_action} to Land")
        available_runways_land = {rid: rdata for rid, rdata in runways_copy_actions.items() if rdata.get('status') == 'Available'}
        selected_runway_land = st.selectbox("Select Landing Runway", options=[""] + sorted(list(available_runways_land.keys())), key=f"rwy_land_{selected_flight_id_action}")
        if st.button("Direct to Land", key=f"land_btn_{selected_flight_id_action}") and selected_runway_land:
             try:
                with data_lock:
                    if selected_flight_id_action in st.session_state.flights and selected_runway_land in st.session_state.runways and st.session_state.runways[selected_runway_land].get('status') == 'Available':
                        flight_data = st.session_state.flights[selected_flight_id_action]; new_runway_data = st.session_state.runways[selected_runway_land]; freed_resources=[]
                        old_rwy = flight_data.get('target_runway') # Free previous runway
                        if old_rwy and old_rwy != selected_runway_land and old_rwy in st.session_state.runways and st.session_state.runways[old_rwy].get('flight_id') == selected_flight_id_action:
                            st.session_state.runways[old_rwy]['status'] = 'Available'; st.session_state.runways[old_rwy]['flight_id'] = None; freed_resources.append(f"Runway {old_rwy}")
                        # --- Calculate Landing Waypoints (Slowdown + Threshold ONLY) ---
                        slowdown_pt = get_runway_slowdown_point(new_runway_data)
                        thresh_pt, _ = calculate_endpoint(new_runway_data['x'], new_runway_data['y'], new_runway_data['length_px'], new_runway_data['angle_deg'])
                        flight_data['waypoints'] = [slowdown_pt, thresh_pt] # Set the two waypoints
                        flight_data['current_waypoint_index'] = 0; flight_data['status'] = STATUS_APPROACHING
                        flight_data['speed'] = APPROACH_SPEED_KNOTS # Start approach fast
                        flight_data['target_runway'] = selected_runway_land; flight_data['takeoff_clearance_time'] = None
                        new_runway_data['status'] = 'Occupied'; new_runway_data['flight_id'] = selected_flight_id_action # Occupy runway
                        msg = f"Directing {selected_flight_id_action} to land on {selected_runway_land}.";
                        if freed_resources: msg += f" ({', '.join(freed_resources)} freed)."
                        st.success(msg); save_data(st.session_state.flights, st.session_state.runways); st.rerun()
                    # (Error checks...)
                    elif selected_flight_id_action not in st.session_state.flights: st.error("Flight not found.")
                    elif selected_runway_land not in st.session_state.runways: st.error("Runway not found.")
                    else: st.error("Runway no longer available.")
             except Exception as e: st.error(f"Error directing landing: {e}")
        st.divider()

        # --- Takeoff Command --- (Unchanged logic)
        if current_status == STATUS_ON_GROUND:
            st.subheader(f"Command {selected_flight_id_action} to Take Off")
            available_runways_dep = {rid: rdata for rid, rdata in runways_copy_actions.items() if rdata.get('status') == 'Available'}
            selected_runway_dep = st.selectbox("Select Departure Runway", options=[""] + sorted(list(available_runways_dep.keys())), key=f"rwy_dep_{selected_flight_id_action}")
            if st.button(f"Prepare Takeoff from {selected_runway_dep}", key=f"takeoff_btn_{selected_flight_id_action}") and selected_runway_dep:
                 try:
                     with data_lock:
                         if selected_flight_id_action in st.session_state.flights and st.session_state.flights[selected_flight_id_action].get('status') == STATUS_ON_GROUND and selected_runway_dep in st.session_state.runways and st.session_state.runways[selected_runway_dep].get('status') == 'Available':
                             flight_data = st.session_state.flights[selected_flight_id_action]; runway_data = st.session_state.runways[selected_runway_dep]
                             runway_data['status'] = 'Occupied'; runway_data['flight_id'] = selected_flight_id_action
                             dep_thresh_pt, _ = get_runway_departure_points(runway_data); dep_x, dep_y = dep_thresh_pt
                             flight_data['status'] = STATUS_PREPARING_TAKEOFF; flight_data['speed'] = 0; flight_data['altitude'] = 0
                             flight_data['x'] = dep_x; flight_data['y'] = dep_y; flight_data['target_runway'] = selected_runway_dep
                             flight_data['takeoff_clearance_time'] = time.time(); flight_data['waypoints'] = []; flight_data['current_waypoint_index'] = -1
                             st.success(f"Flight {selected_flight_id_action} preparing takeoff from {selected_runway_dep} ({TAKEOFF_DELAY_SECONDS}s delay).")
                             save_data(st.session_state.flights, st.session_state.runways); st.rerun()
                         # (Failure cases...)
                         elif selected_flight_id_action not in st.session_state.flights: st.error("Flight not found.")
                         elif st.session_state.flights[selected_flight_id_action].get('status') != STATUS_ON_GROUND: st.error("Flight not On Ground.")
                         elif selected_runway_dep not in st.session_state.runways: st.error("Runway not found.")
                         else: st.error("Runway no longer available.")
                 except Exception as e: st.error(f"Error commanding takeoff: {e}")

    st.divider()
    # --- Display Current Flights Data --- (Unchanged)
    st.header("Current Flights")
    with data_lock: flights_copy_disp = copy.deepcopy(st.session_state.flights)
    if flights_copy_disp:
        display_data = {}
        for fid, data in flights_copy_disp.items():
             display_data[fid] = {'X': int(data.get('x',0)),'Y': int(data.get('y',0)),'Alt(ft)': int(data.get('altitude',0)),'Spd(kts)': int(data.get('speed',0)),'Dir(°)': int(data.get('direction',0)),'Status': data.get('status','Unk'),'Waypts': len(data.get('waypoints',[])),'WP Idx': data.get('current_waypoint_index',-1),'Runway': data.get('target_runway','-')}
        st.dataframe(pd.DataFrame.from_dict(display_data, orient='index'))
        st.subheader("Remove Flight") # Remove flight option
        flight_to_remove = st.selectbox("Select Flight to Remove", options=[""] + sorted(list(flights_copy_disp.keys())), key="remove_select")
        if st.button("Remove Selected Flight") and flight_to_remove:
            with data_lock:
                if flight_to_remove in st.session_state.flights:
                    flight_data = st.session_state.flights[flight_to_remove]; target_runway = flight_data.get('target_runway')
                    if target_runway and target_runway in st.session_state.runways and st.session_state.runways[target_runway].get('flight_id') == flight_to_remove:
                        st.session_state.runways[target_runway]['status'] = 'Available'; st.session_state.runways[target_runway]['flight_id'] = None
                    del st.session_state.flights[flight_to_remove]
                    st.success(f"Flight {flight_to_remove} removed."); save_data(st.session_state.flights, st.session_state.runways)
                else: st.warning(f"Flight {flight_to_remove} not found.")
            st.rerun()
    else: st.info("No flights currently in airspace.")

    # --- Display Infrastructure Status --- (Unchanged)
    st.header("Infrastructure Status")
    st.subheader("Runways")
    with data_lock: runways_copy_infra = copy.deepcopy(st.session_state.runways)
    runways_display = {rid: {'Status': rdata.get('status','?'), 'Flight': rdata.get('flight_id','-'), 'Angle': rdata.get('angle_deg','?')} for rid, rdata in runways_copy_infra.items()}
    st.dataframe(pd.DataFrame.from_dict(runways_display, orient='index'))

    # --- Add Agent Control Button ---
    st.divider()
    st.header("ATC Agent Control")
    # Display status message from session state
    agent_status = st.session_state.get('agent_status_message', 'Agents Idle.')
    st.info(agent_status)

    if st.button("Run ATC Agents Cycle", key="run_agents_btn", disabled=st.session_state.get('agent_cycle_running', False)):
        run_atc_orchestrator(agent_resources) # Pass the cached resources
        st.rerun()


with col1: # Map Column
    st.header("Airspace Map")
    img = np.ones((MAP_HEIGHT_PX, MAP_WIDTH_PX, 3), dtype=np.uint8) * 255
    # --- Draw Runways --- (Unchanged)
    with data_lock: runways_draw_copy = copy.deepcopy(st.session_state.runways)
    for rwy_id, data in runways_draw_copy.items():
        try:
            x_c, y_c = int(data['x']), int(data['y']); angle_deg = data['angle_deg']; length_px = data['length_px']; width_px = data['width_px']
            p1, p2 = calculate_endpoint(x_c, y_c, length_px, angle_deg); cv2.line(img, p1, p2, (100, 100, 100), width_px)
            cv2.putText(img, rwy_id, (x_c + 10, y_c - width_px), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        except Exception as e: print(f"Error drawing runway {rwy_id}: {e}")

    # --- Draw Flights and Waypoints --- (Unchanged logic for hiding/coloring)
    ARROW_LENGTH = 30; WAYPOINT_COLOR = (0, 180, 0); WAYPOINT_LINE_COLOR = (180, 180, 180)
    with data_lock: flights_draw_copy = copy.deepcopy(st.session_state.flights)
    for flight_id, data in flights_draw_copy.items():
        status = data.get('status', 'Unknown')
        if status in [STATUS_ON_GROUND, STATUS_PREPARING_TAKEOFF]: continue # Skip drawing
        try:
            fx, fy = int(data['x']), int(data['y']); direction_deg = data.get('direction', 0.0)
            altitude = data.get('altitude', 0.0); speed = data.get('speed', 0.0)
            waypoints = data.get('waypoints', []); wp_index = data.get('current_waypoint_index', -1)
            # Determine Color
            if altitude <= 5000: flight_color = COLOR_ALT_LOW
            elif altitude <= 15000: flight_color = COLOR_ALT_MEDIUM_LOW
            elif altitude <= 25000: flight_color = COLOR_ALT_MEDIUM_HIGH
            else: flight_color = COLOR_ALT_HIGH
            # Draw Waypoint Path
            if wp_index >= 0 and len(waypoints) > 0:
                if wp_index < len(waypoints): tx, ty = waypoints[wp_index]; cv2.line(img, (fx, fy), (int(tx), int(ty)), WAYPOINT_LINE_COLOR, 1, cv2.LINE_AA)
                for i in range(wp_index, len(waypoints) - 1): x1, y1 = waypoints[i]; x2, y2 = waypoints[i+1]; cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), WAYPOINT_LINE_COLOR, 1, cv2.LINE_AA)
                for i in range(wp_index, len(waypoints)): wp_x, wp_y = waypoints[i]; cv2.circle(img, (int(wp_x), int(wp_y)), 8, WAYPOINT_COLOR, -1)
            # Draw Flight Icon
            if speed > 0.1: # Arrow if moving
                angle_rad = math.radians(90 - direction_deg); x2 = int(fx + ARROW_LENGTH * math.cos(angle_rad)); y2 = int(fy - ARROW_LENGTH * math.sin(angle_rad))
                if 0 <= fx < MAP_WIDTH_PX and 0 <= fy < MAP_HEIGHT_PX and 0 <= x2 < MAP_WIDTH_PX and 0 <= y2 < MAP_HEIGHT_PX: cv2.arrowedLine(img, (fx, fy), (x2, y2), flight_color, 3, tipLength=0.4)
                elif 0 <= fx < MAP_WIDTH_PX and 0 <= fy < MAP_HEIGHT_PX: cv2.circle(img, (fx, fy), 5, flight_color, -1)
            else: # Dot if stopped
                 if 0 <= fx < MAP_WIDTH_PX and 0 <= fy < MAP_HEIGHT_PX: cv2.circle(img, (fx, fy), 7, flight_color, -1)
            # Draw Label
            if 0 <= fx < MAP_WIDTH_PX and 0 <= fy < MAP_HEIGHT_PX:
                 label = f"{flight_id} ({int(altitude/100)})"; cv2.putText(img, label, (fx + 10, fy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        except Exception as e: print(f"Error drawing flight {flight_id}: {e}")
    st.image(img, channels="BGR", use_container_width=True) # Display map
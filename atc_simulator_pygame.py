import pygame
import threading
import time
import math
import pandas as pd # Keep for potential future data display? Or remove if not used.
import numpy as np
import os
import pickle
import copy
import cv2 # Still used for drawing onto the numpy array first
from dotenv import load_dotenv
import ast # <<< ADD IMPORT
import random # <<< ADD IMPORT

# --- LangChain / Groq Imports ---
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.memory import ConversationBufferMemory # Not currently used
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

# --- Configuration ---
DATA_FILE = "atc_state_no_gates_v3.pkl"
MAP_WIDTH_PX = 4000
MAP_HEIGHT_PX = 4000
MAP_SIZE_KM = 50
PIXELS_PER_KM = MAP_WIDTH_PX / MAP_SIZE_KM

# --- Pygame Specific Config ---
WINDOW_WIDTH = 1000 # Adjust as needed for screen size
WINDOW_HEIGHT = 800 # Adjust as needed
FPS = 30 # Frames per second
AGENT_CYCLE_INTERVAL_SECONDS = 15 # How often to run the agent cycle - INCREASED

# --- Simulation Constants (Copied) ---
RUNWAY_LENGTH_KM = 1.2
RUNWAY_WIDTH_PX = 40
WAYPOINT_ARRIVAL_THRESHOLD_PX = 75
APPROACH_SPEED_KNOTS = 160.0
DEFAULT_FLIGHT_SPEED_KNOTS = 250.0
CLIMB_OUT_DISTANCE_KM = 5.0
ALTITUDE_UPDATE_RATE_FPM = 1500.0
TAKEOFF_DELAY_SECONDS = 10.0
LANDING_SLOWDOWN_DISTANCE_KM = 5.0
LANDING_SPEED_KNOTS = 50.0

STATUS_EN_ROUTE = "En Route"
STATUS_APPROACHING = "Approaching"
STATUS_ON_GROUND = "On Ground"
STATUS_PREPARING_TAKEOFF = "Preparing for Takeoff"
STATUS_DEPARTING = "Departing"
STATUS_FOLLOWING_WAYPOINTS = "Following Waypoints"

COLOR_ALT_LOW = (0, 100, 0)
COLOR_ALT_MEDIUM_LOW = (0, 165, 255)
COLOR_ALT_MEDIUM_HIGH = (255, 100, 0)
COLOR_ALT_HIGH = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RUNWAY = (100, 100, 100)
COLOR_RUNWAY_OCCUPIED = (255, 165, 0) # Orange for occupied
COLOR_WAYPOINT = (0, 180, 0)
COLOR_WAYPOINT_LINE = (180, 180, 180)
COLOR_TEXT = (50, 50, 50)
COLOR_BACKGROUND = COLOR_WHITE # Define background color constant

# --- Helper Functions (Copied) ---
def calculate_endpoint(x_center, y_center, length_px, angle_deg):
    angle_rad = math.radians(90 - angle_deg); half_len = length_px / 2.0
    dx = half_len * math.cos(angle_rad); dy = half_len * math.sin(angle_rad)
    x1, y1 = x_center - dx, y_center + dy; x2, y2 = x_center + dx, y_center - dy
    return (int(x1), int(y1)), (int(x2), int(y2))

def get_runway_slowdown_point(runway_data, slowdown_dist_km=LANDING_SLOWDOWN_DISTANCE_KM):
    threshold_pt, _ = calculate_endpoint(runway_data['x'], runway_data['y'], runway_data['length_px'], runway_data['angle_deg'])
    thresh_x, thresh_y = threshold_pt
    approach_angle_deg = (runway_data['angle_deg'] + 180) % 360; approach_angle_rad = math.radians(90 - approach_angle_deg)
    offset_px = slowdown_dist_km * PIXELS_PER_KM
    offset_dx = offset_px * math.cos(approach_angle_rad); offset_dy = offset_px * math.sin(approach_angle_rad)
    slowdown_x = thresh_x + offset_dx; slowdown_y = thresh_y - offset_dy
    return (int(slowdown_x), int(slowdown_y))

def get_runway_departure_points(runway_data, climb_out_dist_km=CLIMB_OUT_DISTANCE_KM):
    _ , departure_threshold_pt = calculate_endpoint(runway_data['x'], runway_data['y'], runway_data['length_px'], runway_data['angle_deg'])
    dep_x, dep_y = departure_threshold_pt
    departure_angle_deg = runway_data['angle_deg']; departure_angle_rad = math.radians(90 - departure_angle_deg)
    offset_px = climb_out_dist_km * PIXELS_PER_KM
    offset_dx = offset_px * math.cos(departure_angle_rad); offset_dy = offset_px * math.sin(departure_angle_rad)
    climb_out_x = dep_x + offset_dx; climb_out_y = dep_y - offset_dy
    return departure_threshold_pt, (int(climb_out_x), int(climb_out_y))


# --- ATC Simulator Class ---
class ATCSimulatorPygame:
    def __init__(self):
        # --- Pygame Init ---
        pygame.init()
        pygame.font.init() # Initialize font module
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("ATC Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30) # Basic font - INCREASED SIZE

        # --- State Init ---
        self.data_lock = threading.Lock() # Still needed if agents run threaded
        self.flights = {}
        self.runways = {}
        self.flight_counter = 0 # <<< ADD Counter for unique IDs
        self.load_data() # Load initial state

        # --- Agent Init ---
        load_dotenv() # Load API Key
        self.agent_resources = self._initialize_agents_and_llm()
        self.agent_cycle_running = False
        self.last_agent_run_time = time.time()
        self.agent_status_message = "Agents Idle."
        self.last_random_flight_check_time = time.time() # <<< ADD Timer

        # --- Simulation Timing ---
        self.last_update_time = time.time()

        # --- Background Sim Update Thread ---
        # Separate thread for physics updates to keep rendering smooth
        self.simulation_running = True
        self.sim_update_thread = threading.Thread(target=self._simulation_update_loop, daemon=True)
        self.sim_update_thread.start()

    def load_data(self):
        """Loads simulation state from file or initializes defaults."""
        # Adapted from Streamlit version, uses self.flights/runways
        if os.path.exists(DATA_FILE):
            try:
                with self.data_lock: # Protect read during load
                     with open(DATA_FILE, "rb") as f:
                        data = pickle.load(f)
                        self.flights = data.get('flights', {})
                        self.runways = data.get('runways', {})
                        print("Loaded data from file.")
                        # Default checks (same as before)
                        default_runway_length_px = RUNWAY_LENGTH_KM * PIXELS_PER_KM
                        for rwy_data in self.runways.values():
                             rwy_data.setdefault('angle_deg', 90.0); rwy_data.setdefault('length_px', default_runway_length_px)
                             rwy_data.setdefault('width_px', RUNWAY_WIDTH_PX); rwy_data.setdefault('status', 'Available'); rwy_data.setdefault('flight_id', None)
                        for flt_data in self.flights.values():
                            flt_data.setdefault('waypoints', []); flt_data.setdefault('current_waypoint_index', -1)
                            flt_data.setdefault('status', STATUS_EN_ROUTE); flt_data.setdefault('target_runway', None)
                            flt_data.setdefault('altitude', 30000.0); flt_data.setdefault('speed', DEFAULT_FLIGHT_SPEED_KNOTS)
                            flt_data.setdefault('direction', 90.0); flt_data.setdefault('takeoff_clearance_time', None)
                        return # Success
            except Exception as e:
                print(f"Error loading data file: {e}. Initializing fresh state.")
        else:
            print("No data file found. Initializing fresh state.")
        # Default initial state if load fails or no file
        runway_length_px = RUNWAY_LENGTH_KM * PIXELS_PER_KM
        self.flights = {}
        self.runways = { "RW27L": {"status": "Available", "flight_id": None, "x": 2000, "y": 3800, "angle_deg": 270.0, "length_px": runway_length_px, "width_px": RUNWAY_WIDTH_PX}, "RW09R": {"status": "Available", "flight_id": None, "x": 2000, "y": 3000, "angle_deg": 90.0, "length_px": runway_length_px, "width_px": RUNWAY_WIDTH_PX} }

    def save_data(self):
        """Saves the current simulation state."""
        # Use self.data_lock to ensure safe copy
        with self.data_lock:
            data_to_save = {
                'flights': copy.deepcopy(self.flights),
                'runways': copy.deepcopy(self.runways)
            }
        try:
            with open(DATA_FILE, "wb") as f:
                pickle.dump(data_to_save, f)
                print("Data saved.")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _parse_tool_input(self, tool_input, expected_args_count):
        """Safely parses tool input, handling strings or list/tuples."""
        print(f"DEBUG TOOL Parse: Raw Input='{tool_input}', Type={type(tool_input)}")
        args = None
        if isinstance(tool_input, (list, tuple)):
            args = tool_input
        elif isinstance(tool_input, str):
            try:
                # Strip potential outer quotes if the agent adds them
                if tool_input.startswith("'") and tool_input.endswith("'"):
                    tool_input = tool_input[1:-1]
                elif tool_input.startswith('"') and tool_input.endswith('"'):
                    tool_input = tool_input[1:-1]

                args = ast.literal_eval(tool_input)
                if not isinstance(args, (list, tuple)):
                    print(f"ERROR TOOL Parse: Evaluated string is not list/tuple: {args}")
                    # Attempt to re-parse if it looks like a nested string representation
                    if isinstance(args, str) and (args.startswith('(') or args.startswith('[')): # Check for list/tuple start
                         print(f"DEBUG TOOL Parse: Attempting re-parse of nested string...")
                         args = ast.literal_eval(args)
                         if not isinstance(args, (list, tuple)):
                              raise ValueError("Re-parsed string did not evaluate to a list/tuple.")
                    else:
                         raise ValueError("Input string did not evaluate to a list/tuple.")

                print(f"DEBUG TOOL Parse: Evaluated string to: {args}")
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"ERROR TOOL Parse: Failed to parse input string: {e}")
                # Raise the error so the agent framework can handle it
                raise ValueError(f"Invalid string input format for tool: '{tool_input}'. Error: {e}") from e
        else:
            # Raise the error so the agent framework can handle it
            raise TypeError(f"Unexpected input type for tool: {type(tool_input)}")

        # Ensure args is now a list/tuple before checking length
        if not isinstance(args, (list, tuple)):
             print(f"ERROR TOOL Parse: Args is not a list/tuple after parsing: {type(args)}")
             raise TypeError(f"Parsed arguments are not a list/tuple: {type(args)}")

        if len(args) != expected_args_count:
             print(f"ERROR TOOL Parse: Incorrect argument count. Expected {expected_args_count}, Got {len(args)}")
             # Raise the error
             raise ValueError(f"Incorrect argument count. Expected {expected_args_count}, Got {len(args)}")

        return args

    # --- Agent Tool Functions (as Methods) ---
    # These now access self.flights, self.runways, self.data_lock
    def get_all_aircraft_info(self):
        with self.data_lock: return copy.deepcopy(self.flights)
    def get_all_runway_info(self):
        with self.data_lock: return copy.deepcopy(self.runways)
    def get_distances_to_runways(self):
        """Calculates distance from each active flight to each runway threshold."""
        print("DEBUG TOOL GetDistancesToRunways: Calculating distances...")
        distances = {}
        try:
            with self.data_lock:
                # Create copies to work with safely
                flights_copy = copy.deepcopy(self.flights)
                runways_copy = copy.deepcopy(self.runways)

            active_flights = {
                fid: data for fid, data in flights_copy.items()
                if data.get('status') not in [STATUS_ON_GROUND, STATUS_PREPARING_TAKEOFF]
                   and 'x' in data and 'y' in data
            }

            if not active_flights:
                print("DEBUG TOOL GetDistancesToRunways: No active flights.")
                return {"info": "No active flights to calculate distances for."}
            if not runways_copy:
                 print("DEBUG TOOL GetDistancesToRunways: No runways defined.")
                 return {"info": "No runways defined to calculate distances to."}


            for flight_id, flight_data in active_flights.items():
                flight_x, flight_y = flight_data['x'], flight_data['y']
                distances[flight_id] = {}
                for runway_id, runway_data in runways_copy.items():
                    try:
                        # Calculate landing threshold point (p1)
                        threshold_pt, _ = calculate_endpoint(
                            runway_data['x'], runway_data['y'],
                            runway_data['length_px'], runway_data['angle_deg']
                        )
                        thresh_x, thresh_y = threshold_pt
                        # Calculate Euclidean distance
                        dist_px = math.sqrt((thresh_x - flight_x)**2 + (thresh_y - flight_y)**2)
                        dist_km = dist_px / PIXELS_PER_KM
                        # Store distance in km for relevance
                        distances[flight_id][runway_id] = round(dist_km, 1)
                    except Exception as e_dist:
                         print(f"ERROR TOOL GetDistances: Failed for {flight_id} to {runway_id}: {e_dist}")
                         distances[flight_id][runway_id] = "Error"

            print(f"DEBUG TOOL GetDistancesToRunways: Calculated: {distances}")
            return distances

        except Exception as e_tool:
            print(f"ERROR TOOL GetDistancesToRunways: General error: {e_tool}")
            return {"error": f"Internal error in GetDistancesToRunways: {e_tool}"}

    def set_waypoints(self, flight_id: str, waypoints: list[tuple[int, int]]):
        print(f"DEBUG TOOL SetWaypoints: Received flight_id='{flight_id}', waypoints={waypoints}") # <<< ADD DEBUG
        try: # <<< ADD TRY
            # Validation logic remains the same
            if not isinstance(flight_id, str) or not flight_id: return {"error": "Invalid flight_id."}
            if not isinstance(waypoints, list): return {"error": "Waypoints must be a list."}
            valid_waypoints = []
            for wp in waypoints:
                if isinstance(wp, (list, tuple)) and len(wp) == 2:
                    try:
                        wp_x, wp_y = int(wp[0]), int(wp[1])
                        if 0 <= wp_x < MAP_WIDTH_PX and 0 <= wp_y < MAP_HEIGHT_PX: valid_waypoints.append((wp_x, wp_y))
                        else: return {"error": f"Wpt {wp} out of bounds."}
                    except (ValueError, TypeError): return {"error": f"Invalid wpt format: {wp}."}
                else: return {"error": f"Invalid wpt format: {wp}."}
            if not valid_waypoints: return {"info": "No valid waypoints provided."}

            with self.data_lock:
                if flight_id not in self.flights: return {"error": f"Flight {flight_id} not found."}
                flight_data = self.flights[flight_id]
                msg_prefix = f"Waypoints set for {flight_id}."; freed_resources = []
                old_rwy = flight_data.get('target_runway')
                if old_rwy and old_rwy in self.runways and self.runways[old_rwy].get('flight_id') == flight_id:
                    self.runways[old_rwy]['status'] = 'Available'; self.runways[old_rwy]['flight_id'] = None; freed_resources.append(f"Rwy {old_rwy}")
                flight_data['waypoints'] = valid_waypoints; flight_data['current_waypoint_index'] = 0; flight_data['status'] = STATUS_FOLLOWING_WAYPOINTS
                flight_data['speed'] = DEFAULT_FLIGHT_SPEED_KNOTS; flight_data['target_runway'] = None; flight_data['takeoff_clearance_time'] = None
                if freed_resources: msg_prefix += f" ({', '.join(freed_resources)} freed)."
                print(f"INFO (Tool SetWaypoints): {msg_prefix}")
            return {"success": msg_prefix}
        except Exception as e_tool: # <<< ADD EXCEPT
            print(f"ERROR TOOL SetWaypoints for {flight_id}: {e_tool}")
            return {"error": f"Internal error in SetWaypoints: {e_tool}"}

    def initiate_landing(self, flight_id: str, runway_id: str):
        print(f"DEBUG TOOL InitiateLanding: Received flight_id='{flight_id}', runway_id='{runway_id}'") # <<< ADD DEBUG
        try: # <<< ADD TRY
            if not isinstance(flight_id, str) or not flight_id: return {"error": "Invalid flight_id."}
            if not isinstance(runway_id, str) or not runway_id: return {"error": "Invalid runway_id."}
            with self.data_lock:
                if flight_id not in self.flights: return {"error": f"Flight {flight_id} not found."}
                if runway_id not in self.runways: return {"error": f"Runway {runway_id} not found."}
                if self.runways[runway_id].get('status') != 'Available': return {"error": f"Runway {runway_id} not available."}
                status = self.flights[flight_id].get('status')
                if status in [STATUS_ON_GROUND, STATUS_PREPARING_TAKEOFF, STATUS_DEPARTING]: return {"error": f"Flight {flight_id} cannot land (Status: {status})."}

                flight_data = self.flights[flight_id]; new_runway_data = self.runways[runway_id]; freed_resources=[]
                old_rwy = flight_data.get('target_runway')
                if old_rwy and old_rwy != runway_id and old_rwy in self.runways and self.runways[old_rwy].get('flight_id') == flight_id:
                    self.runways[old_rwy]['status'] = 'Available'; self.runways[old_rwy]['flight_id'] = None; freed_resources.append(f"Rwy {old_rwy}")
                slowdown_pt = get_runway_slowdown_point(new_runway_data)
                thresh_pt, _ = calculate_endpoint(new_runway_data['x'], new_runway_data['y'], new_runway_data['length_px'], new_runway_data['angle_deg'])
                flight_data['waypoints'] = [slowdown_pt, thresh_pt]; flight_data['current_waypoint_index'] = 0; flight_data['status'] = STATUS_APPROACHING
                flight_data['speed'] = APPROACH_SPEED_KNOTS; flight_data['target_runway'] = runway_id; flight_data['takeoff_clearance_time'] = None
                new_runway_data['status'] = 'Occupied'; new_runway_data['flight_id'] = flight_id
                msg = f"Directing {flight_id} to land on {runway_id}.";
                if freed_resources: msg += f" ({', '.join(freed_resources)} freed)."
                print(f"INFO (Tool InitiateLanding): {msg}")
            return {"success": msg}
        except Exception as e_tool: # <<< ADD EXCEPT
            print(f"ERROR TOOL InitiateLanding for {flight_id} on {runway_id}: {e_tool}")
            return {"error": f"Internal error in InitiateLanding: {e_tool}"}

    def initiate_takeoff(self, flight_id: str, runway_id: str):
        print(f"DEBUG TOOL InitiateTakeoff: Received flight_id='{flight_id}', runway_id='{runway_id}'") # <<< ADD DEBUG
        try: # <<< ADD TRY
            if not isinstance(flight_id, str) or not flight_id: return {"error": "Invalid flight_id."}
            if not isinstance(runway_id, str) or not runway_id: return {"error": "Invalid runway_id."}
            with self.data_lock:
                if flight_id not in self.flights: return {"error": f"Flight {flight_id} not found."}
                if runway_id not in self.runways: return {"error": f"Runway {runway_id} not found."}
                if self.flights[flight_id].get('status') != STATUS_ON_GROUND: return {"error": f"Flight {flight_id} not On Ground."}
                if self.runways[runway_id].get('status') != 'Available': return {"error": f"Runway {runway_id} not available."}

                flight_data = self.flights[flight_id]; runway_data = self.runways[runway_id]
                runway_data['status'] = 'Occupied'; runway_data['flight_id'] = flight_id
                dep_thresh_pt, _ = get_runway_departure_points(runway_data); dep_x, dep_y = dep_thresh_pt
                flight_data['status'] = STATUS_PREPARING_TAKEOFF; flight_data['speed'] = 0; flight_data['altitude'] = 0
                flight_data['x'] = float(dep_x); flight_data['y'] = float(dep_y); flight_data['target_runway'] = runway_id
                flight_data['takeoff_clearance_time'] = time.time(); flight_data['waypoints'] = []; flight_data['current_waypoint_index'] = -1
                msg = f"Flight {flight_id} preparing takeoff from {runway_id} ({TAKEOFF_DELAY_SECONDS}s delay)."
                print(f"INFO (Tool InitiateTakeoff): {msg}")
            return {"success": msg}
        except Exception as e_tool: # <<< ADD EXCEPT
             print(f"ERROR TOOL InitiateTakeoff for {flight_id} from {runway_id}: {e_tool}")
             return {"error": f"Internal error in InitiateTakeoff: {e_tool}"}

    # --- Agent Initialization (as Method) ---
    def _initialize_agents_and_llm(self):
        """Initializes the LLM, RAG components, tools, and agents."""
        # ACCEPT that this runs every time the script starts for now
        print("--- Running Full Agent & RAG Initialization ---")
        initialized_resources = {
            "llm": None, "tools": [], "scheduler_agent": None,
            "conflict_agent": None, "retriever": None # Add retriever
        }
        # 1. Initialize LLM
        try:
            llm = ChatGroq(temperature=0.1, model_name="llama3-70b-8192")
            print("Groq LLM Initialized.")
            initialized_resources["llm"] = llm
        except Exception as e:
            print(f"ERROR: Failed to initialize Groq LLM: {e}")
            self.agent_status_message = f"LLM Init Failed: {e}"
            return initialized_resources # LLM is critical

        # --- 2. Initialize RAG Pipeline ---
        sop_file = "SOP.md"
        if os.path.exists(sop_file):
            print(f"Loading SOP document: {sop_file}")
            try:
                # Load MARKDOWN instead of PDF
                loader = UnstructuredMarkdownLoader(sop_file)
                docs = loader.load()

                # Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                print(f"SOP split into {len(splits)} chunks.")

                # Create Embeddings (Downloads model on first run)
                print("Initializing embedding model...")
                # Using a common, relatively small sentence transformer
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
                print("Embedding model loaded.")

                # Create FAISS Vector Store (In-Memory)
                print("Creating FAISS vector store from SOP chunks...")
                vectorstore = FAISS.from_documents(splits, hf_embeddings)
                print("FAISS vector store created.")

                # Create Retriever
                retriever = vectorstore.as_retriever()
                initialized_resources["retriever"] = retriever
                print("SOP Retriever Initialized.")

            except Exception as e_rag:
                print(f"ERROR during RAG pipeline setup: {e_rag}")
                self.agent_status_message = f"RAG Init Failed: {e_rag}"
                # Continue without retriever if RAG fails? Or fail completely?
                # Let's continue without it for now, comms will fallback.
        else:
            print(f"WARNING: {sop_file} not found. Communication agent will not use RAG.")
            self.agent_status_message = "SOP.md not found; using basic comms."


        # --- 3. Define Tools ---
        current_tools = [
            Tool(name="GetAllAircraftInfo", func=lambda _: self.get_all_aircraft_info(), description="Gets current state of all aircraft."),
            Tool(name="GetAllRunwayInfo", func=lambda _: self.get_all_runway_info(), description="Gets current state of all runways."),
            Tool(name="GetDistancesToRunways",
                 func=lambda _: self.get_distances_to_runways(),
                 description="Calculates and returns the distance (in km) from each active flight to the landing threshold of every runway. Useful for prioritizing landings."),
            Tool(name="SetWaypoints",
                 func=lambda tool_input: self.set_waypoints(*self._parse_tool_input(tool_input, 2)),
                 description="Sets waypoints. Input MUST be tuple/list string like '(\"FLT123\", [(100, 200)])' or the actual list/tuple.",
                 handle_tool_error=True),
            Tool(name="InitiateLanding",
                 func=lambda tool_input: self.initiate_landing(*self._parse_tool_input(tool_input, 2)),
                 description="Directs landing. Input MUST be tuple/list string like '(\"FLT123\", \"RW27L\")' or the actual list/tuple.",
                 handle_tool_error=True),
            Tool(name="InitiateTakeoff",
                 func=lambda tool_input: self.initiate_takeoff(*self._parse_tool_input(tool_input, 2)),
                 description="Commands takeoff. Input MUST be tuple/list string like '(\"FLT123\", \"RW27L\")' or the actual list/tuple.",
                 handle_tool_error=True),
        ]
        print(f"LangChain Tools Defined: {[tool.name for tool in current_tools]}")
        initialized_resources["tools"] = current_tools

        # --- 4. Initialize Agents ---
        llm = initialized_resources.get("llm")
        if not llm:
             print("ERROR: LLM not available for agent initialization.")
             return initialized_resources

        # -- Scheduler Agent (Using full prompt with distance tool mention) --
        scheduler_prompt_template = """You are an expert Air Traffic Control Scheduler. Your primary responsibility is to manage landings and takeoffs efficiently and safely using the available runways and airspace.

**Your Goal:** Decide the single most important scheduling action for this cycle: assigning a landing clearance, a takeoff clearance, or vectoring an aircraft into a holding pattern if necessary.

**Available Tools:**
{tools}

**Current Airspace State:**
{input}

**Your Thought Process (Chain-of-Thought):**
1.  **Analyze State:** Review the current positions, altitudes, statuses, and destinations of all aircraft. Check the status and assignments of all runways.
2.  **Check Distances:** Use `GetDistancesToRunways` to understand the proximity of active flights to potential landing runways.
3.  **Identify Needs & Opportunities:**
    *   Are any flights `En Route` or `Following Waypoints` and close enough to a runway threshold (use distance info) to be considered for landing?
    *   Are any flights `On Ground` and ready for departure?
    *   Are all relevant runways available or occupied?
4.  **Prioritize Landing:** Landing aircraft generally have priority. If multiple aircraft are suitable for landing on available runways, prioritize the one closest to its respective runway threshold (based on `GetDistancesToRunways`).
5.  **Decide Action (Max ONE per cycle):**
    *   **If Landing Possible:** Identify the highest priority landing candidate and an available runway. Issue an `InitiateLanding` command.
    *   **If No Urgent Landing & Takeoff Ready:** If no flights require immediate landing instructions *and* there's a flight `On Ground` *and* a suitable runway is `Available`, issue an `InitiateTakeoff` command.
    *   **If Congestion/Waiting:** If a flight needs to land but its target runway (or the only suitable one) is occupied, or if multiple aircraft are converging, consider using `SetWaypoints` to assign a *single* holding waypoint near the aircraft's current position to keep it clear of traffic.
    *   **If No Action:** If no landings are feasible, no takeoffs are ready, and no holding patterns are needed, state that no action is required.
6.  **Tool Usage:** Use tools precisely as described. Input for tools taking multiple arguments **MUST** be a Python tuple or list provided *as a single string argument* to the tool, like `'("FLT123", "RW27L")'` or `'["FLT456", [(1000, 1500)]]'`. **Do NOT call multiple tools.**

**Examples of Usage:**

*   **Example 1: Landing Assignment**
    *   `Input Scenario:` Flights: { "UAL101": {"status": "En Route", "x": 1500, "y": 3500, "altitude": 5000}, "AAL202": {"status": "En Route", "x": 3000, "y": 1000, "altitude": 28000} }, Runways: { "RW27L": {"status": "Available", "x": 2000, "y": 3800}, "RW09R": {"status": "Available", "x": 2000, "y": 3000} }, Distances: { "UAL101": {"RW27L": 8.5, "RW09R": 25.0}, "AAL202": {"RW27L": 30.0, "RW09R": 22.0} }
    *   `Thought:` 1. UAL101 is close to RW27L, AAL202 is further away. Both runways available. 2. Distances confirm UAL101 is much closer to RW27L (8.5km). 3. Landing needed for UAL101. 4. Prioritize UAL101 for RW27L. 5. Issue landing command.
    *   `Action:` `InitiateLanding('("UAL101", "RW27L")')`

*   **Example 2: Takeoff Assignment**
    *   `Input Scenario:` Flights: { "DAL303": {"status": "On Ground", "x": 1400, "y": 2980}, "SWA404": {"status": "En Route", "x": 3500, "y": 3500, "altitude": 30000} }, Runways: { "RW27L": {"status": "Available"}, "RW09R": {"status": "Available"} }, Distances: { "SWA404": {"RW27L": 25.0, "RW09R": 15.0} }
    *   `Thought:` 1. DAL303 is On Ground. SWA404 is far away. Both runways available. 2. No immediate landing need. 3. DAL303 is ready for takeoff. RW09R is likely the departure runway based on position, and it's available. 4. Schedule takeoff.
    *   `Action:` `InitiateTakeoff('("DAL303", "RW09R")')`

*   **Example 3: Holding Pattern Needed**
    *   `Input Scenario:` Flights: { "ACA505": {"status": "En Route", "x": 2500, "y": 3600, "altitude": 6000}, "BAW606": {"status": "Approaching", "target_runway": "RW27L", "x": 1000, "y": 3800, "altitude": 2000} }, Runways: { "RW27L": {"status": "Occupied", "flight_id": "BAW606"}, "RW09R": {"status": "Available"} }, Distances: { "ACA505": {"RW27L": 6.0, "RW09R": 20.0}, "BAW606": {"RW27L": 1.0, "RW09R": 15.0} }
    *   `Thought:` 1. ACA505 is close to RW27L but BAW606 is already approaching it. RW27L is occupied. 2. ACA505 cannot land now. 3. Need to keep ACA505 clear. 4. Assign a holding waypoint nearby, e.g., slightly north-east at (2800, 3400). 5. Issue waypoint command.
    *   `Action:` `SetWaypoints('("ACA505", [(2800, 3400)])')`

**Constraint Reminder:** You can only issue ONE command (InitiateLanding, InitiateTakeoff, or SetWaypoints) OR state that no action is needed. Ensure tool inputs are correctly formatted as single strings representing lists/tuples.

Begin!

{agent_scratchpad}
""" # Use full prompt text
        SCHEDULER_PROMPT = ChatPromptTemplate.from_messages([
            ("system", scheduler_prompt_template),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]) # Use full prompt structure
        try:
            scheduler_agent = initialize_agent(tools=current_tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, prompt=SCHEDULER_PROMPT, verbose=True, handle_parsing_errors=True)
            print("Scheduler Agent Initialized.")
            initialized_resources["scheduler_agent"] = scheduler_agent
        except Exception as e: print(f"Error initializing Scheduler Agent: {e}")

        # -- Conflict Resolution Agent (Using full prompt) --
        conflict_prompt_template = """You are an expert Air Traffic Control Conflict Resolution specialist. Your sole focus is identifying and resolving potential losses of separation between airborne aircraft.

**Your Goal:** Detect if any pair of *active* aircraft (NOT `On Ground` or `Preparing Takeoff`) are predicted to violate minimum separation rules (less than 150 pixels horizontally AND less than 1000 feet vertically). If a conflict exists, issue a command to resolve it, prioritizing safety and minimizing disruption.

**Minimum Safe Separation:** 150 pixels horizontally **AND** 1000 feet vertically. A conflict exists only if *both* horizontal and vertical separation minima are breached or predicted to be breached soon.

**Available Tools:**
{tools}

**Current Airspace State:**
{input}

**Your Thought Process (Chain-of-Thought):**
1.  **Identify Active Flights:** List all flights whose status is NOT `On Ground` or `Preparing Takeoff`. If less than two active flights, stop (no conflicts possible).
2.  **Analyze Pairs:** For every possible pair of active flights (A, B):
    *   Calculate current horizontal distance in pixels between A and B.
    *   Calculate current vertical distance in feet between A and B.
    *   **Predict Conflict:** Check if they are currently closer than 150px horizontally AND closer than 1000ft vertically. (Focus on *current* separation for simplicity).
3.  **Conflict Detected?** If a pair (A, B) violates *both* horizontal and vertical separation minima:
    *   **Prioritize Resolution:** This is a conflict. You must act. Handle only ONE conflict per cycle (the first one detected or the most critical if multiple).
    *   **Check Statuses:** Look at the `status` of both conflicting aircraft (A and B).
    *   **Determine Who to Vector:**
        *   If one aircraft (e.g., A) has status `Approaching`, **DO NOT** change its waypoints. You **MUST** vector the other aircraft (B).
        *   If neither aircraft is `Approaching`, you can choose one to vector (e.g., the one slightly higher, or based on direction, but a simple choice is often sufficient).
    *   **Calculate Vector:** For the chosen aircraft (let's say B), calculate a *single* new waypoint designed to increase separation. A simple strategy is a 30-45 degree turn away from the other aircraft (A) for a short distance (e.g., 5km ahead on the new heading). Calculate the (x, y) coordinates for this single waypoint. Example: If B is at (1000, 1000) and needs to turn right, a waypoint like (1200, 800) might be suitable.
    *   **Issue Command:** Use the `SetWaypoints` tool for the chosen aircraft (B) with the single calculated waypoint `[(x, y)]`.
    *   **Stop:** Do not look for further conflicts in this cycle.
4.  **No Conflict?** If, after checking all pairs, no aircraft are violating *both* horizontal and vertical separation, state clearly that no conflicts were detected.

**Tool Usage:** ONLY use `SetWaypoints` if a conflict is detected and resolution is needed. Input **MUST** be a Python tuple or list provided *as a single string argument* to the tool, like `'("FLT789", [(1200, 800)])'`.

**Examples of Usage:**

*   **Example 1: Simple Conflict Resolution**
    *   `Input Scenario:` Active Flights: { "JBU111": {"status": "En Route", "x": 1500, "y": 1500, "altitude": 10000}, "EJA222": {"status": "Following Waypoints", "x": 1550, "y": 1550, "altitude": 10500} }
    *   `Thought:` 1. Active: JBU111, EJA222. 2. Pair (JBU111, EJA222): Horiz dist = sqrt(50^2+50^2) = 71px. Vert dist = 500ft. 3. Conflict Check: 71px < 150px AND 500ft < 1000ft. YES, conflict! 4. Resolution: Neither is Approaching. Choose to vector EJA222. Turn EJA222 right slightly. Calculate waypoint, e.g., (1700, 1400). 5. Issue command.
    *   `Action:` `SetWaypoints('("EJA222", [(1700, 1400)])')`

*   **Example 2: Conflict with Approaching Aircraft**
    *   `Input Scenario:` Active Flights: { "NKS333": {"status": "Approaching", "target_runway": "RW09R", "x": 1000, "y": 3000, "altitude": 2500}, "WJA444": {"status": "En Route", "x": 1100, "y": 2950, "altitude": 3000} }
    *   `Thought:` 1. Active: NKS333, WJA444. 2. Pair (NKS333, WJA444): Horiz dist = sqrt(100^2+(-50)^2) = sqrt(12500) = 111px. Vert dist = 500ft. 3. Conflict Check: 111px < 150px AND 500ft < 1000ft. YES, conflict! 4. Resolution: NKS333 is `Approaching`. **MUST NOT vector NKS333.** Must vector WJA444. Turn WJA444 away (e.g., south-east). Calculate waypoint, e.g., (1300, 2800). 5. Issue command.
    *   `Action:` `SetWaypoints('("WJA444", [(1300, 2800)])')`

*   **Example 3: No Conflict Found**
    *   `Input Scenario:` Active Flights: { "FDX555": {"status": "En Route", "x": 500, "y": 500, "altitude": 31000}, "UPS666": {"status": "En Route", "x": 550, "y": 550, "altitude": 33000} }
    *   `Thought:` 1. Active: FDX555, UPS666. 2. Pair (FDX555, UPS666): Horiz dist = 71px. Vert dist = 2000ft. 3. Conflict Check: 71px < 150px is TRUE, BUT 2000ft < 1000ft is FALSE. Condition (Both TRUE) is not met. 4. No conflict.
    *   `Final Answer:` No conflicts detected requiring action.

**Constraint Reminder:** Resolve only ONE conflict per cycle. **NEVER** issue `SetWaypoints` for an aircraft with status `Approaching` if it's involved in a conflict; vector the *other* aircraft instead. Ensure tool inputs are correctly formatted as single strings representing lists/tuples.

Begin!

{agent_scratchpad}
""" # Use full prompt text
        CONFLICT_PROMPT = ChatPromptTemplate.from_messages([
            ("system", conflict_prompt_template),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]) # Use full prompt structure
        try:
            conflict_agent = initialize_agent(tools=current_tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, prompt=CONFLICT_PROMPT, verbose=True, handle_parsing_errors=True)
            print("Conflict Resolution Agent Initialized.")
            initialized_resources["conflict_agent"] = conflict_agent
        except Exception as e: print(f"Error initializing Conflict Agent: {e}")


        print("--- Full Initialization Complete ---")
        return initialized_resources

    # --- Simulation Update Logic (threaded) ---
    def _simulation_update_loop(self):
        """Background loop to update flight physics."""
        while self.simulation_running:
            current_time = time.time()
            time_delta_sec = current_time - self.last_update_time
            if time_delta_sec < 0.01: # Avoid excessive updates if loop is too fast
                time.sleep(0.01)
                continue
            self.last_update_time = current_time

            # --- Update flight positions based on time_delta_sec ---
            LANDING_SLOWDOWN_DISTANCE_PX = LANDING_SLOWDOWN_DISTANCE_KM * PIXELS_PER_KM
            updates_to_apply = {}
            flights_to_remove = []
            alt_change_per_sec = ALTITUDE_UPDATE_RATE_FPM / 60.0

            with self.data_lock:
                current_flights_copy = copy.deepcopy(self.flights)
                current_runways_copy = copy.deepcopy(self.runways)

            for flight_id, data in current_flights_copy.items():
                status = data.get('status', 'Unknown')
                final_updates = {} # Store changes for this flight

                # --- Handle Preparing for Takeoff Timer ---
                if status == STATUS_PREPARING_TAKEOFF:
                    takeoff_time = data.get('takeoff_clearance_time')
                    if takeoff_time and time.time() >= takeoff_time + TAKEOFF_DELAY_SECONDS:
                        print(f"DEBUG {flight_id}: Takeoff delay over. Starting departure roll.")
                        prep_updates = {'status': STATUS_DEPARTING, 'speed': DEFAULT_FLIGHT_SPEED_KNOTS, 'takeoff_clearance_time': None}
                        target_runway_id = data.get('target_runway')
                        if target_runway_id and target_runway_id in current_runways_copy:
                            runway_data = current_runways_copy[target_runway_id]
                            _ , climb_out_pt = get_runway_departure_points(runway_data)
                            prep_updates['waypoints'] = [climb_out_pt]; prep_updates['current_waypoint_index'] = 0
                            dep_thresh_x, dep_thresh_y = data['x'], data['y'] # Use current position at threshold
                            climb_dx = climb_out_pt[0]-dep_thresh_x; climb_dy = climb_out_pt[1]-dep_thresh_y
                            if climb_dx != 0 or climb_dy != 0:
                                 angle_rad = math.atan2(-climb_dy, climb_dx); angle_deg = math.degrees(angle_rad)
                                 prep_updates['direction'] = (90 - angle_deg + 360) % 360
                        else: prep_updates['status'] = STATUS_EN_ROUTE # Failsafe if runway gone
                        final_updates.update(prep_updates)
                    # Store updates and skip other processing for this tick
                    if final_updates: updates_to_apply[flight_id] = final_updates
                    continue

                # --- Skip if On Ground ---
                if status == STATUS_ON_GROUND: continue
                if 'x' not in data or 'y' not in data: continue # Skip if invalid

                # --- Process Moving/Active Flights ---
                current_x, current_y = data['x'], data['y']
                speed_knots = data.get('speed', 0.0)
                current_direction = data.get('direction', 0.0)
                waypoints = data.get('waypoints', [])
                wp_index = data.get('current_waypoint_index', -1)
                current_alt = data.get('altitude', 0.0)
                target_runway_id = data.get('target_runway')
                calculated_direction = current_direction
                is_approaching = (status == STATUS_APPROACHING)
                runway_released_this_tick = False # Flag to prevent double release

                # --- Landing Speed Adjustment ---
                if is_approaching and target_runway_id and target_runway_id in current_runways_copy:
                    runway_data = current_runways_copy[target_runway_id]
                    threshold_pt, _ = calculate_endpoint(runway_data['x'], runway_data['y'], runway_data['length_px'], runway_data['angle_deg'])
                    dist_to_thresh = math.sqrt((threshold_pt[0] - current_x)**2 + (threshold_pt[1] - current_y)**2)
                    if dist_to_thresh <= LANDING_SLOWDOWN_DISTANCE_PX and speed_knots > LANDING_SPEED_KNOTS:
                         final_updates['speed'] = LANDING_SPEED_KNOTS
                         speed_knots = LANDING_SPEED_KNOTS # Use updated speed

                # --- Calculate distance covered THIS TICK (regardless of waypoints) ---
                dist_covered_this_tick = 0.0 # Default
                if speed_knots > 0.1: # Only calculate if moving significantly
                    speed_km_s = (speed_knots * 1.852) / 3600
                    dist_covered_this_tick = (speed_km_s * PIXELS_PER_KM) * time_delta_sec

                # --- Waypoint Navigation ---
                if wp_index >= 0 and wp_index < len(waypoints):
                    target_x, target_y = waypoints[wp_index]
                    dx = target_x - current_x; dy = target_y - current_y
                    distance_to_target = math.sqrt(dx**2 + dy**2)

                    # Simplified arrival threshold check
                    speed_km_s = (speed_knots * 1.852) / 3600
                    dist_covered_this_tick = (speed_km_s * PIXELS_PER_KM) * time_delta_sec

                    # Check if arrival threshold is crossed OR distance covered is more than remaining distance
                    if distance_to_target <= WAYPOINT_ARRIVAL_THRESHOLD_PX or (dist_covered_this_tick >= distance_to_target and speed_knots > 0.1):
                        # --- LANDING THRESHOLD ARRIVAL ---
                        if is_approaching and wp_index == len(waypoints) - 1:
                            print(f"DEBUG LANDING: {flight_id} reached landing threshold for runway {target_runway_id}")
                            final_updates['status'] = STATUS_ON_GROUND
                            final_updates['speed'] = 0.0
                            final_updates['altitude'] = 0.0
                            final_updates['waypoints'] = []
                            final_updates['current_waypoint_index'] = -1
                            final_updates['x'] = float(target_x)
                            final_updates['y'] = float(target_y) # Snap position
                            
                            # This is critical - make sure we carry the runway ID for release before clearing it
                            if target_runway_id:  # Store runway ID for release before clearing reference
                                runway_released_this_tick = True
                                final_updates['target_runway'] = None  # Clear target runway only after setting flag
                                updates_to_apply.setdefault(target_runway_id, {})['release_runway'] = flight_id
                                print(f"DEBUG LANDING: Marking runway {target_runway_id} for release by {flight_id}")
                            else:
                                print(f"WARNING: Flight {flight_id} landed but had no target_runway_id!")
                                
                            speed_knots = 0 # Prevent movement calc below
                            print(f"DEBUG {flight_id}: Landed on {target_runway_id}. All landing updates prepared.")
                        # --- REGULAR WAYPOINT ARRIVAL ---
                        else:
                            final_updates['current_waypoint_index'] = wp_index + 1
                            wp_index += 1
                            if wp_index >= len(waypoints): # Reached final regular waypoint
                                final_updates['waypoints'] = []; final_updates['current_waypoint_index'] = -1
                                if status == STATUS_DEPARTING:
                                    final_updates['status'] = STATUS_EN_ROUTE; final_updates['speed'] = DEFAULT_FLIGHT_SPEED_KNOTS
                                    final_updates['target_runway'] = None # Clear target runway
                                    runway_released_this_tick = True
                                    print(f"DEBUG {flight_id}: Reached climb-out, En Route from {target_runway_id}.")
                                elif status == STATUS_FOLLOWING_WAYPOINTS:
                                    final_updates['status'] = STATUS_EN_ROUTE
                                    print(f"DEBUG {flight_id}: Finished waypoints, En Route.")
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
                            new_direction = (90 - angle_deg + 360) % 360
                            # Only update if direction changed significantly to avoid jitter
                            if abs(new_direction - current_direction) > 0.1:
                                 final_updates['direction'] = new_direction
                            calculated_direction = new_direction # Use for movement calculation

                # --- Position Update Calculation ---
                new_x, new_y = current_x, current_y
                if speed_knots > 0:
                    # Use calculated_direction which might have been updated by waypoint logic
                    move_angle_rad = math.radians(90 - calculated_direction)
                    delta_x = dist_covered_this_tick * math.cos(move_angle_rad)
                    delta_y = dist_covered_this_tick * math.sin(move_angle_rad)
                    new_x = current_x + delta_x
                    new_y = current_y - delta_y # Screen coordinates

                # --- MAP EXIT CHECK ---
                if new_x < 0 or new_x >= MAP_WIDTH_PX or new_y < 0 or new_y >= MAP_HEIGHT_PX:
                    print(f"INFO: Flight {flight_id} exited map boundaries. Removing.")
                    flights_to_remove.append(flight_id)
                    runway_released_this_tick = True # Ensure runway released if it exits while assigned
                    continue # Skip other updates

                # --- Apply Position Update ---
                if abs(new_x - current_x) > 0.01: final_updates.setdefault('x', max(0.0, min(new_x, MAP_WIDTH_PX - 1.0)))
                if abs(new_y - current_y) > 0.01: final_updates.setdefault('y', max(0.0, min(new_y, MAP_HEIGHT_PX - 1.0)))

                # --- Altitude Update ---
                if status == STATUS_DEPARTING:
                     final_updates['altitude'] = min(35000.0, current_alt + alt_change_per_sec * time_delta_sec)
                elif status == STATUS_APPROACHING:
                     final_updates['altitude'] = max(0.0, current_alt - alt_change_per_sec * time_delta_sec)

                # Store updates for this flight
                if final_updates:
                    if flight_id in updates_to_apply: updates_to_apply[flight_id].update(final_updates)
                    else: updates_to_apply[flight_id] = final_updates

                # --- Release Runway if needed ---
                # Done separately after all updates for the flight are decided
                if runway_released_this_tick and target_runway_id:
                    # Queue runway release to be applied in the main lock later
                    updates_to_apply.setdefault(target_runway_id, {})['release_runway'] = flight_id


            # --- Apply all updates and removals (within the main lock) ---
            if updates_to_apply or flights_to_remove:
                with self.data_lock:
                    runways_modified = False
                    # First pass: Collect runway releases
                    runways_to_release = {}  # Separate runway release tracking
                    for entity_id, updates in updates_to_apply.items():
                        release_info = updates.get('release_runway')
                        if release_info is not None:
                            # If this is a runway ID with release info
                            if entity_id in self.runways:
                                runways_to_release[entity_id] = release_info
                    
                    # Second pass: Apply flight updates
                    for entity_id, updates in updates_to_apply.items():
                        if entity_id in self.flights:
                            # This is a flight update
                            updates_copy = updates.copy()
                            updates_copy.pop('release_runway', None)  # Remove release instruction if present
                            self.flights[entity_id].update(updates_copy)
                    
                    # Third pass: Process runway releases
                    for runway_id, flight_id in runways_to_release.items():
                        if runway_id in self.runways and self.runways[runway_id].get('flight_id') == flight_id:
                            # Release the runway if it's still assigned to this flight
                            self.runways[runway_id]['status'] = 'Available'
                            self.runways[runway_id]['flight_id'] = None
                            print(f"DEBUG: Runway {runway_id} released by {flight_id} (landing completed).")
                            runways_modified = True
                        else:
                            print(f"WARN: Could not release runway {runway_id} for flight {flight_id} (State mismatch)")

                    # Remove flights
                    for flight_id in flights_to_remove:
                        if flight_id in self.flights:
                            # Check and release any assigned runway for removed flight
                            target_runway = self.flights[flight_id].get('target_runway')
                            if target_runway and target_runway in self.runways and self.runways[target_runway].get('flight_id') == flight_id:
                                self.runways[target_runway]['status'] = 'Available'
                                self.runways[target_runway]['flight_id'] = None
                                print(f"INFO: Runway {target_runway} released due to flight {flight_id} removal.")
                            del self.flights[flight_id]
                            print(f"INFO: Flight {flight_id} removed from simulation.")

            # Short sleep to prevent CPU hogging if updates are very fast
            time.sleep(0.01) # Sleep even if no updates happened

    # --- Agent Orchestration (THREAD TARGET) ---
    def _agent_cycle_thread_target(self):
        """Target function for the background agent execution thread."""
        # This function contains the logic previously in run_agent_cycle
        print("\n--- Background Agent Cycle Started ---")
        # Get resources safely within the thread
        llm = self.agent_resources.get("llm")
        scheduler_agent = self.agent_resources.get("scheduler_agent")
        conflict_agent = self.agent_resources.get("conflict_agent")
        retriever = self.agent_resources.get("retriever")

        if not llm or not scheduler_agent or not conflict_agent:
            comm_message = "ERROR: Agents not initialized."
            print("AgentThread: ERROR - Required agent resources not initialized.")
            with self.data_lock: # Use lock to update shared state
                 self.agent_status_message = comm_message
                 self.agent_cycle_running = False # Mark as finished (due to error)
            return

        action_details = {}
        action_taken_this_cycle = False
        comm_message = "Agents running..." # Initial status for thread

        try:
            # 1. Get State
            current_aircraft_info = self.get_all_aircraft_info()
            current_runway_info = self.get_all_runway_info()
            if not current_aircraft_info:
                 comm_message = "No aircraft; agents idle."
                 print("AgentThread: No aircraft detected, exiting.")
                 # Need to update status and running flag before returning
                 with self.data_lock:
                     self.agent_status_message = comm_message
                     self.agent_cycle_running = False
                 return

            current_state_summary = f"Time: {time.time():.0f}\nAircraft: {current_aircraft_info}\nRunways: {current_runway_info}"
            print(f"AgentThread: State Snapshot:\n{current_state_summary}")

            # 2. Run Scheduler
            print("AgentThread: Running SCHEDULER...")
            scheduler_input_dict = {"input": current_state_summary}
            try:
                scheduler_response = scheduler_agent.invoke(scheduler_input_dict)
                scheduler_output = scheduler_response.get('output', '')
                print(f"Scheduler Raw Output: {scheduler_output}")
                # --- Parsing logic (same as before) ---
                if "InitiateLanding successfully" in scheduler_output or "Successfully directed" in scheduler_output:
                    parts = scheduler_output.split()
                    try: flight_id=parts[parts.index("landing")-1]; runway_id=parts[parts.index("on")+1].replace('.',''); action_details[flight_id]=f"cleared approach {runway_id}"; action_taken_this_cycle=True
                    except: action_details["Scheduler"]="Landed (parse fail)"; action_taken_this_cycle=True
                elif "InitiateTakeoff successfully" in scheduler_output or "preparing for takeoff" in scheduler_output:
                    parts = scheduler_output.split()
                    try: flight_id=parts[parts.index("takeoff")-1]; runway_id=parts[parts.index("from")+1].replace('.',''); action_details[flight_id]=f"cleared takeoff {runway_id}"; action_taken_this_cycle=True
                    except: action_details["Scheduler"]="Took off (parse fail)"; action_taken_this_cycle=True
                elif "SetWaypoints successfully" in scheduler_output or "Waypoints set for" in scheduler_output:
                    parts = scheduler_output.split()
                    try: flight_id=parts[parts.index("for")+1].replace('.',''); action_details[flight_id]="assigned waypoints"; action_taken_this_cycle=True
                    except: action_details["Scheduler"]="Set WPs (parse fail)"; action_taken_this_cycle=True
            except Exception as e: print(f"ERROR running Scheduler: {e}")

            # 3. Run Conflict Resolver
            print("\nAgentThread: Running CONFLICT RESOLVER...")
            current_aircraft_info = self.get_all_aircraft_info() # Re-get state
            current_runway_info = self.get_all_runway_info()
            active_flights_count = sum(1 for d in current_aircraft_info.values() if d.get('status') not in [STATUS_ON_GROUND, STATUS_PREPARING_TAKEOFF])
            if active_flights_count >= 2:
                conflict_state_summary = f"Time: {time.time():.0f}\nAircraft: {current_aircraft_info}\nRunways: {current_runway_info}"
                conflict_input_dict = {"input": conflict_state_summary}
                try:
                    conflict_response = conflict_agent.invoke(conflict_input_dict)
                    conflict_output = conflict_response.get('output', '')
                    print(f"Conflict Raw Output: {conflict_output}")
                    # --- Parsing (same basic logic) ---
                    if "SetWaypoints successfully" in conflict_output or "Waypoints set for" in conflict_output:
                        parts = conflict_output.split()
                        try:
                             flight_id=parts[parts.index("for")+1].replace('.','')
                             if flight_id not in action_details: action_details[flight_id]="vector for traffic"; action_taken_this_cycle=True
                        except:
                            if "Conflict" not in action_details: action_details["Conflict"]="Vector (parse fail)"; action_taken_this_cycle=True
                    elif "No conflicts detected" in conflict_output: print("AgentThread: No conflicts reported.")
                except Exception as e: print(f"ERROR running Conflict Resolver: {e}")
            else: print("AgentThread: Skipping conflict check.")

            # --- 4. Generate Communication (using RAG if available) ---
            if action_details:
                # ... (Generate comms logic using RAG/fallback as before) ...
                # Store result in local comm_message variable
                print("\nAgentThread: Generating Communication...")
                question = "Generate concise, realistic ATC radio calls based on standard phraseology for the following actions:\n"
                question += "\n".join([f"- {f}: {d}" for f, d in action_details.items()])

                if retriever: # Use RAG pipeline
                    # ... (RAG chain setup and invoke) ...
                    template = """Answer the question based only on the following context, simulating standard ATC phraseology. If the context isn't relevant, generate a standard concise radio call.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:"""
                    prompt = ChatPromptTemplate.from_template(template)
                    rag_chain = (
                        RunnableParallel(
                           {"context": retriever, "question": RunnablePassthrough()}
                        )
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    try:
                        final_comm = rag_chain.invoke(question)
                        print(f"RAG Comms Raw Output: {final_comm}")
                        comm_message = "RAG Comms:\n" + final_comm
                    except Exception as e_rag_invoke:
                         print(f"ERROR invoking RAG chain: {e_rag_invoke}")
                         comm_message = "RAG Comm Failed."
                         try:
                             comm_resp = llm.invoke(question + "\n\nAnswer:")
                             comm_message = "Comms (RAG fallback):\n" + getattr(comm_resp, 'content', str(comm_resp))
                         except Exception as e_llm_fb: print(f"ERROR basic LLM fallback: {e_llm_fb}")
                else: # Fallback to basic LLM call
                    print("AgentThread: Using basic LLM for communication.")
                    try:
                        comm_resp = llm.invoke(question + "\n\nAnswer:")
                        comm_message = "Comms (Basic):\n" + getattr(comm_resp, 'content', str(comm_resp))
                    except Exception as e_llm_basic: print(f"ERROR basic LLM: {e_llm_basic}"); comm_message = "Basic Comm Failed."
            else:
                comm_message = "Agents finished: No actions taken."

            # --- 5. Save State --- 
            if action_taken_this_cycle:
                print("\nAgentThread: Saving state...")
                self.save_data() # save_data uses its own lock

        except Exception as e_cycle:
            print(f"ERROR in background agent cycle: {e_cycle}")
            comm_message = f"Agent Cycle Failed: {e_cycle}"

        # --- Update shared state AFTER cycle completes or fails --- 
        with self.data_lock:
             self.agent_status_message = comm_message
             self.agent_cycle_running = False # Reset the flag

        print("--- Background Agent Cycle Finished ---")

    # --- Drawing ---
    def draw(self):
        """Draws the simulation state onto the screen."""
        # 1. Create Base Map Image with Background Color (Full Size)
        map_img = np.full((MAP_HEIGHT_PX, MAP_WIDTH_PX, 3), COLOR_BACKGROUND, dtype=np.uint8)

        # 2. Get safe copies of state for drawing
        with self.data_lock:
            runways_draw_copy = copy.deepcopy(self.runways)
            flights_draw_copy = copy.deepcopy(self.flights)

        # 3. Draw Runways onto the large map_img
        for rwy_id, data in runways_draw_copy.items():
            try:
                x_c, y_c = int(data['x']), int(data['y']); angle = data['angle_deg']
                l_px, w_px = data['length_px'], data['width_px']
                p1, p2 = calculate_endpoint(x_c, y_c, l_px, angle)
                # --- Select color based on status --- 
                rwy_color = COLOR_RUNWAY_OCCUPIED if data.get('status') == 'Occupied' else COLOR_RUNWAY
                cv2.line(map_img, p1, p2, rwy_color, w_px)
                cv2.putText(map_img, rwy_id, (x_c + 10, y_c - w_px), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)
            except Exception as e: print(f"Error drawing runway {rwy_id}: {e}")

        # 4. Draw Flights and Waypoints onto the large map_img
        ARROW_LENGTH = 30
        for flight_id, data in flights_draw_copy.items():
            status = data.get('status', 'Unknown')
            if status in [STATUS_ON_GROUND, STATUS_PREPARING_TAKEOFF]: continue # Skip drawing these

            try:
                fx, fy = int(data['x']), int(data['y']); direction = data.get('direction', 0.0)
                alt = data.get('altitude', 0.0); speed = data.get('speed', 0.0)
                waypoints = data.get('waypoints', []); wp_idx = data.get('current_waypoint_index', -1)

                # Determine Color
                if alt <= 5000: color = COLOR_ALT_LOW
                elif alt <= 15000: color = COLOR_ALT_MEDIUM_LOW
                elif alt <= 25000: color = COLOR_ALT_MEDIUM_HIGH
                else: color = COLOR_ALT_HIGH

                # Draw Waypoint Path
                if wp_idx >= 0 and len(waypoints) > 0:
                    if wp_idx < len(waypoints):
                        tx, ty = waypoints[wp_idx]
                        cv2.line(map_img, (fx, fy), (int(tx), int(ty)), COLOR_WAYPOINT_LINE, 1, cv2.LINE_AA)
                    for i in range(wp_idx, len(waypoints) - 1):
                        x1, y1 = waypoints[i]; x2, y2 = waypoints[i+1]
                        cv2.line(map_img, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_WAYPOINT_LINE, 1, cv2.LINE_AA)
                    for i in range(wp_idx, len(waypoints)):
                        wp_x, wp_y = waypoints[i]
                        cv2.circle(map_img, (int(wp_x), int(wp_y)), 8, COLOR_WAYPOINT, -1)

                # Draw Flight Icon
                if speed > 0.1: # Arrow if moving
                    angle_rad = math.radians(90 - direction)
                    x2 = int(fx + ARROW_LENGTH * math.cos(angle_rad))
                    y2 = int(fy - ARROW_LENGTH * math.sin(angle_rad))
                    # Basic boundary check for arrow drawing
                    if 0 <= fx < MAP_WIDTH_PX and 0 <= fy < MAP_HEIGHT_PX:
                         cv2.arrowedLine(map_img, (fx, fy), (x2, y2), color, 3, tipLength=0.4)
                    else: # Draw dot if arrow goes offscreen
                         cv2.circle(map_img, (fx, fy), 5, color, -1)
                else: # Dot if stopped
                     if 0 <= fx < MAP_WIDTH_PX and 0 <= fy < MAP_HEIGHT_PX:
                         cv2.circle(map_img, (fx, fy), 7, color, -1)

                # Draw Label - UPDATED FORMAT
                if 0 <= fx < MAP_WIDTH_PX and 0 <= fy < MAP_HEIGHT_PX:
                    label = f"{flight_id} ({int(alt/100)} {int(speed)}kts)" # Added speed
                    cv2.putText(map_img, label, (fx + 10, fy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            except Exception as e: print(f"Error drawing flight {flight_id}: {e}")

        # --- 5. Scale the ENTIRE map down to window size ---
        try:
            scaled_map_img = cv2.resize(map_img, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)
        except Exception as e_resize:
            # ...(handle resize error)
            return

        # 6. Convert SCALED image for Pygame Display
        rgb_scaled_np = cv2.cvtColor(scaled_map_img, cv2.COLOR_BGR2RGB)

        # 7. Transpose axes for Pygame (width, height)
        transposed_rgb_scaled = np.transpose(rgb_scaled_np, (1, 0, 2))

        # 8. Create Pygame surface from the TRANSPOSED, SCALED array
        try:
             pygame_surface = pygame.surfarray.make_surface(transposed_rgb_scaled)
             # Keep orientation fix if needed (depends on exact libraries/versions)
             # pygame_surface = pygame.transform.flip(pygame_surface, False, True) # Add back if needed
        except Exception as e_surf:
            print(f"ERROR creating Pygame surface: {e_surf}")
             # ...(handle surface error)

        # 9. Blit the final surface to Screen at (0,0)
        self.screen.blit(pygame_surface, (0, 0))

        # 10. Draw Agent Status Text (on top)
        status_surface = self.font.render(self.agent_status_message, True, COLOR_BLACK, COLOR_BACKGROUND)
        self.screen.blit(status_surface, (10, 10))

    # --- Main Loop ---
    def run(self):
        """Main application loop."""
        running = True
        BACKGROUND_COLOR = COLOR_BACKGROUND

        while running:
            current_loop_time = time.time()
            # --- Handle Input ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Quit event received. Stopping simulation...")
                    running = False
                    self.simulation_running = False # Signal sim thread to stop
                # Add keyboard handling here later if needed (e.g., run agents manually)

            # --- Random Flight Check (approx every minute) ---
            if current_loop_time - self.last_random_flight_check_time > 15.0:
                print("Checking for random flight addition...")
                if random.randint(1, 8) == 1:
                    self._add_random_flight()
                self.last_random_flight_check_time = current_loop_time # Reset timer

            # --- Agent Cycle Check --- 
            # Use the flag self.agent_cycle_running read safely
            with self.data_lock:
                 is_agent_running = self.agent_cycle_running

            if not is_agent_running and (current_loop_time - self.last_agent_run_time > AGENT_CYCLE_INTERVAL_SECONDS):
                 print("MainLoop: Triggering agent cycle in background thread...")
                 with self.data_lock: # Safely set the running flag
                      self.agent_cycle_running = True
                      self.agent_status_message = "Agents starting..." # Update status immediately

                 # --- Start agent cycle in a background thread --- 
                 agent_thread = threading.Thread(target=self._agent_cycle_thread_target, daemon=True)
                 agent_thread.start()
                 self.last_agent_run_time = current_loop_time # Reset timer now

            # --- Drawing ---
            # self.screen.fill(BACKGROUND_COLOR) # REMOVED - Background is part of draw() now
            self.draw() # Draw handles its own background now
            pygame.display.flip()

            # --- Timing ---
            self.clock.tick(FPS)

        # --- Cleanup ---
        if self.sim_update_thread.is_alive():
            self.sim_update_thread.join(timeout=1.0) # Wait briefly for thread
        pygame.quit()
        print("Application Exited.")

    def _add_random_flight(self):
        """Creates and adds a new flight with random parameters."""
        self.flight_counter += 1
        flight_id = f"RAND{self.flight_counter:03d}"
        print(f"Attempting to add random flight: {flight_id}")

        # Determine entry edge and initial parameters
        edge = random.choice(["top", "bottom", "left", "right"])
        altitude = random.randint(250, 350) * 100.0 # 25k-35k ft
        speed = DEFAULT_FLIGHT_SPEED_KNOTS
        status = STATUS_EN_ROUTE
        x, y, direction = 0.0, 0.0, 0.0

        margin = 50 # Start slightly inside the map
        map_center_x = MAP_WIDTH_PX / 2
        map_center_y = MAP_HEIGHT_PX / 2

        if edge == "top":
            x = random.uniform(margin, MAP_WIDTH_PX - margin)
            y = float(margin)
            # Aim towards general center area south of start
            target_y = random.uniform(map_center_y, MAP_HEIGHT_PX * 0.75)
            angle_rad = math.atan2(-(target_y - y), map_center_x - x)
            direction = (90 - math.degrees(angle_rad) + 360) % 360
        elif edge == "bottom":
            x = random.uniform(margin, MAP_WIDTH_PX - margin)
            y = float(MAP_HEIGHT_PX - margin)
            target_y = random.uniform(MAP_HEIGHT_PX * 0.25, map_center_y)
            angle_rad = math.atan2(-(target_y - y), map_center_x - x)
            direction = (90 - math.degrees(angle_rad) + 360) % 360
        elif edge == "left":
            x = float(margin)
            y = random.uniform(margin, MAP_HEIGHT_PX - margin)
            target_x = random.uniform(map_center_x, MAP_WIDTH_PX * 0.75)
            angle_rad = math.atan2(-(map_center_y - y), target_x - x)
            direction = (90 - math.degrees(angle_rad) + 360) % 360
        elif edge == "right":
            x = float(MAP_WIDTH_PX - margin)
            y = random.uniform(margin, MAP_HEIGHT_PX - margin)
            target_x = random.uniform(MAP_WIDTH_PX * 0.25, map_center_x)
            angle_rad = math.atan2(-(map_center_y - y), target_x - x)
            direction = (90 - math.degrees(angle_rad) + 360) % 360

        new_flight_data = {
            'speed': speed,
            'direction': direction,
            'x': x,
            'y': y,
            'altitude': altitude,
            'status': status,
            'waypoints': [],
            'current_waypoint_index': -1,
            'target_runway': None,
            'takeoff_clearance_time': None
        }

        try:
            with self.data_lock:
                if flight_id not in self.flights:
                    self.flights[flight_id] = new_flight_data
                    print(f"Successfully added random flight {flight_id} at ({int(x)}, {int(y)}), heading {int(direction)}deg")
                else:
                    print(f"WARN: Random flight ID {flight_id} already existed? Skipping.")
        except Exception as e:
            print(f"ERROR: Failed to add random flight {flight_id}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure dependencies are installed:
    # pip install pygame python-dotenv langchain langchain-groq langchain-community pypdf faiss-cpu sentence-transformers tiktoken unstructured
    # Ensure SOP.md exists in the same directory
    simulator = ATCSimulatorPygame()
    simulator.run()
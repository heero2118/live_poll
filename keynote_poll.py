import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import numpy as np

st.set_page_config('📋 Polling Questions', initial_sidebar_state='collapsed')
st.title("📋 Live Poll Demo")
st.divider()

# set up parameters
api_base_url = "https://api.surveysparrow.com/v3" # Change if needed
api_key = st.secrets['access_token']
survey_id = st.secrets['survey_id']
image_url = 'QR-velocity.png'
survey_url = 'https://sprw.io/stt-SHVpi'

# --- API Call Function ---
def fetch_survey_data(base_url, key, s_id, retries=3):
    """Fetches survey responses from the Survey Sparrow API."""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # Endpoint for fetching responses (check V3 documentation for accuracy)
    endpoint = f"{base_url}/responses"
    params = {
        "survey_id": s_id,
        "preserve_format": True,
        "order_by": "completedTime",
        "order": 'ASC'
    }

    for attempt in range(retries):
        try:
            # st.write(f"Attempt {attempt + 1}: Calling API: {endpoint} with survey_id={s_id}")
            response = requests.get(endpoint, headers=headers, params=params, timeout=5) # Added timeout

            # Check for successful response
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            # st.success(f"API call successful (Status code: {response.status_code})")
            return response.json() # Return parsed JSON data

        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP Error occurred: {http_err}")
            st.error(f"Response status code: {response.status_code}")
            try:
                # Try to get more details from the response body if available
                error_details = response.json()
                st.error(f"Error details: {json.dumps(error_details, indent=2)}")
            except json.JSONDecodeError:
                st.error(f"Response content: {response.text}") # Show raw text if not JSON

            if response.status_code == 401:
                 st.error("Authentication failed. Please check your API Key.")
                 return None # Don't retry on auth errors
            elif response.status_code == 404:
                 st.error("Resource not found. Check Survey ID and API endpoint.")
                 return None # Don't retry on not found errors
            # Can add more specific status code handling here
            if attempt == retries - 1: # If last attempt failed
                 return None

        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"Connection Error occurred: {conn_err}")
            if attempt == retries - 1: return None

        except requests.exceptions.Timeout as timeout_err:
            st.error(f"Timeout Error occurred: {timeout_err}")
            if attempt == retries - 1: return None

        except requests.exceptions.RequestException as req_err:
            st.error(f"An unexpected error occurred during the API request: {req_err}")
            if attempt == retries - 1: return None
        except json.JSONDecodeError:
            st.error("Failed to decode API response (not valid JSON).")
            st.text(f"Raw response text:\n{response.text[:500]}...") # Show beginning of raw text
            return None
        except Exception as e:
             st.error(f"An unexpected error occurred: {e}")
             return None

    return None # Return None if all retries fail

# --- Data Processing Function ---
def list_of_json_to_dataframe(response: list) -> pd.DataFrame:
    # write a function to convert response json to dataframe, exclude metadata
    answers_data = {}
    list_json_data = response

    for response_id, json_data in enumerate(list_json_data):
        answers_data[response_id] = {}
        for answer_obj, list_answers in json_data.items():
            if answer_obj == 'answers':
                for answer in list_answers:
                    if ('answer' in answer) and ('question' in answer):
                        answers_data[response_id][answer['question']] = answer['answer']
    df = pd.DataFrame(answers_data).T
    return df

# pull survey result from API and convert to a dataframe
data = fetch_survey_data(api_base_url, api_key, survey_id)
df = list_of_json_to_dataframe(data['data']) # Convert JSON to DataFrame

# map quesion response to scores
string_to_number_x = {
    'Stuck in a Downward Spiral': 1.0,
    'Riding the Revenue Rollercoaster ': 2.0,
    'Winging It on Gut Instinct': 3.0,
    'Mission Control for Measurable Success': 4.0
}
string_to_number_y = {
    'Revolving Door': 1.0,
    'A Means to an End': 2.0,
    'Untapped Potential': 3.0,
    'Talent Magnet': 4.0,
}

# --- 2. Define the Mapping Function ---
def map_text_to_code(text, mapping_dict, default_value=np.nan):
    # Handle potential non-string inputs (like NaN or None)
    if not isinstance(text, str):
        return default_value
    # Iterate through the map items (key_text, numeric_value)
    for key_text, numeric_value in mapping_dict.items():
        # Check if the lowercase text contains the lowercase key_text
        if key_text.lower() in text.lower():
            score_variance = 0.1
            return numeric_value * np.random.uniform(1.0-score_variance, 1.0+score_variance, 1)[0] # Return the corresponding number if found, add some variance
    # If the loop completes without finding any key_text in the input text
    return default_value

df['x_values'] = df['Which of the following best describes the performance of your company?'].apply( lambda y: map_text_to_code(y, string_to_number_x) )
df['y_values'] = df['Which statement best describes how employees feel about working at your company?'].apply( lambda x: map_text_to_code(x, string_to_number_y) )
df['labels'] = df['Name'] # use name as labels


# survey link and QR code
col1, col2 = st.columns([5,5])
with col1:
    st.markdown(f"<h4 style='text-align: center; color: black;'>Scan the QR code or use the link below to access the survey</h4>", unsafe_allow_html=True)
    st.image(image_url)
    survey_link = survey_url
    st.markdown(f"<h4 style='text-align: center; color: blue;'>{survey_link}</h4>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<h4 style='text-align: center; color: black;'># responses:</h4>", unsafe_allow_html=True)
    st.markdown("""
        <style>
        .big-font {
            font-size:100px !important;
            text-align: center
        }
        </style>
        """, unsafe_allow_html=True)
    no_responses = df.shape[0]
    st.markdown(f'<p class="big-font">{no_responses}</p>', unsafe_allow_html=True)
    st.button('Refresh', key='refresh_button', help="Click to refresh the data from the API.", use_container_width=True,)

st.divider()
############################### CHART SECTION START #########################
#############################################################################
# --- Set Default Axis Ranges (with padding) ---
x_min_data = 0
x_max_data = 5
y_min_data = 0
y_max_data = 5

# Add a 10% buffer (adjust buffer as needed)
x_range_buffer = (x_max_data - x_min_data) * 0.1
y_range_buffer = (y_max_data - y_min_data) * 0.1

# Set default min/max slightly outside the data range
default_x_min = x_min_data - x_range_buffer
default_x_max = x_max_data + x_range_buffer
default_y_min = y_min_data - y_range_buffer
default_y_max = y_max_data + y_range_buffer

# Default thresholds (e.g., median or mean of data)
default_x_threshold = 2.5
default_y_threshold = 2.5

# Ensure min is less than max even if buffer is zero or negative (e.g., single point)
if default_x_min >= default_x_max:
    default_x_min -= 1
    default_x_max += 1
if default_y_min >= default_y_max:
    default_y_min -= 1
    default_y_max += 1

# --- 2. Streamlit UI for Customization ---
st.sidebar.header("Chart Customization")

# Chart Size
st.sidebar.subheader("Figure Dimensions")
use_container_width_toggle = st.sidebar.checkbox("Use Container Width", True) # Default to custom size

figure_width = st.sidebar.number_input(
    "Figure Width (pixels)",
    min_value=100,
    max_value=2000,
    value=700,  # Default width
    step=50,
    disabled=use_container_width_toggle # Disable if using container width
    )
figure_height = st.sidebar.number_input(
    "Figure Height (pixels)",
    min_value=100,
    max_value=2000,
    value=600, # Default height
    step=50,
    # Height is independent of container width, so not disabled
    )

# Chart Title
chart_title = st.sidebar.text_input("Chart Title", "Velocity Matrix")
title_font_size = st.sidebar.slider("Title Font Size", 10, 40, 30)
title_font_color = st.sidebar.color_picker("Title Font Color", "#1E9488")

# Axis Labels
x_axis_label = st.sidebar.text_input("X-Axis Label", "Business Performance")
y_axis_label = st.sidebar.text_input("Y-Axis Label", "Company Culture")
axis_label_font_size = st.sidebar.slider("Axis Label Font Size", 8, 30, 20)
axis_label_font_color = st.sidebar.color_picker("Axis Label Font Color", "#1E9488")

# Tick Labels
show_x_tick_labels = st.sidebar.checkbox("Show X-Axis Tick Labels", False)
show_y_tick_labels = st.sidebar.checkbox("Show Y-Axis Tick Labels", False)
tick_label_font_size = st.sidebar.slider("Tick Label Font Size", 6, 25, 12)
tick_label_font_color = st.sidebar.color_picker("Tick Label Font Color", "#666666") #666666 black

# Axis Range Controls
st.sidebar.subheader("Axis Ranges")
x_axis_min = st.sidebar.number_input("X-Axis Min", value=float(default_x_min), step=1.0, format="%.2f")
x_axis_max = st.sidebar.number_input("X-Axis Max", value=float(default_x_max), step=1.0, format="%.2f")
y_axis_min = st.sidebar.number_input("Y-Axis Min", value=float(default_y_min), step=1.0, format="%.2f")
y_axis_max = st.sidebar.number_input("Y-Axis Max", value=float(default_y_max), step=1.0, format="%.2f")
if x_axis_min >= x_axis_max: st.sidebar.warning("X-Axis Min must be less than X-Axis Max.")
if y_axis_min >= y_axis_max: st.sidebar.warning("Y-Axis Min must be less than Y-Axis Max.")

# Axis Lines and Grid
st.sidebar.subheader("Axis & Grid Lines")
show_x_axis_line = st.sidebar.checkbox("Show X-Axis Line", True)
show_y_axis_line = st.sidebar.checkbox("Show Y-Axis Line", True)
axis_line_color = st.sidebar.color_picker("Axis Line Color", "#cccccc")
show_grid = st.sidebar.checkbox("Show Grid", False)
grid_color = st.sidebar.color_picker("Grid Color", "#e0e0e0")

# Quadrant Controls ---
st.sidebar.subheader("Quadrants")
enable_quadrants = st.sidebar.checkbox("Enable Quadrants", True)

if enable_quadrants:
    x_threshold = st.sidebar.number_input("X Threshold", value=default_x_threshold, step=0.5, format="%.2f")
    y_threshold = st.sidebar.number_input("Y Threshold", value=default_y_threshold, step=0.5, format="%.2f")

    st.sidebar.markdown("---") # Separator
    st.sidebar.markdown("**Quadrant Lines**")
    quadrant_line_color = st.sidebar.color_picker("Line Color", "#aaaaaa") # Medium grey
    quadrant_line_width = st.sidebar.slider("Line Width", 1, 5, 1)
    quadrant_line_dash = st.sidebar.selectbox("Line Style", ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'], index=2) # Default to dash

    st.sidebar.markdown("---") # Separator
    st.sidebar.markdown("**Quadrant Labels**")
    label_top_right = st.sidebar.text_input("Top-Right Label", "Optimized")
    label_top_left = st.sidebar.text_input("Top-Left Label", "Inefficient")
    label_bottom_left = st.sidebar.text_input("Bottom-Left Label", "Dysfunctional")
    label_bottom_right = st.sidebar.text_input("Bottom-Right Label", "Unsustainable")
    quadrant_label_font_size = st.sidebar.slider("Label Font Size", 8, 30, 18)
    quadrant_label_font_color = st.sidebar.color_picker("Label Font Color", "#888888") # Lighter grey

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quadrant Background Colors**")
    # st.sidebar.caption("Tip: Use RGBA for transparency, e.g., `rgba(0, 0, 255, 0.1)`")
    color_top_right = st.sidebar.color_picker("Top-Right Color", "#D5F5E3") # Light Green
    color_top_left = st.sidebar.color_picker("Top-Left Color", "#F1F1F1")  # Light Grey
    color_bottom_left = st.sidebar.color_picker("Bottom-Left Color", "#FADBD8") # Light Red
    color_bottom_right = st.sidebar.color_picker("Bottom-Right Color", "#F1F1F1") # Light Grey

# Point and Point Labels
st.sidebar.subheader("Point Labels")
score_variance = st.sidebar.slider("Point Size", 1, 10, 1)/100
point_color = st.sidebar.color_picker("Point Color", "#333333") #FA7031 orange
point_size = st.sidebar.slider("Point Size", 5, 20, 12)
show_point_labels = st.sidebar.checkbox("Show Point Labels", True)
point_label_font_size = st.sidebar.slider("Point Label Font Size", 5, 20, 16)
point_label_font_color = st.sidebar.color_picker("Point Label Font Color", "#333333") #FA7031 orange

# Legend
st.sidebar.subheader("Legend")
show_legend = st.sidebar.checkbox("Show Legend", False)
legend_title = st.sidebar.text_input("Legend Title", "Category")
legend_font_size = st.sidebar.slider("Legend Font Size", 8, 25, 12)
legend_title_font_size = st.sidebar.slider("Legend Title Font Size", 9, 28, 14)


# --- 3. Create the Scatter Plot ---
fig = px.scatter(
    df,
    x='x_values',
    y='y_values',
    text='labels' if show_point_labels else None,
    title=chart_title,
    labels={
        'x_values': x_axis_label,
        'y_values': y_axis_label,
        'category': legend_title
    },
)

# --- 4. Apply Customizations ---
# Update overall layout properties
fig.update_layout(
    width=figure_width if not use_container_width_toggle else None,
    height=figure_height, # Height is applied regardless of container width toggle
    title_font_size=title_font_size,
    title_font_color=title_font_color,
    title_x=0,
    xaxis_title_font_size=axis_label_font_size,
    xaxis_title_font_color=axis_label_font_color,
    yaxis_title_font_size=axis_label_font_size,
    yaxis_title_font_color=axis_label_font_color,
    xaxis_tickfont_size=tick_label_font_size,
    xaxis_tickfont_color=tick_label_font_color,
    yaxis_tickfont_size=tick_label_font_size,
    yaxis_tickfont_color=tick_label_font_color,
    showlegend=show_legend,
    legend_title_font_size=legend_title_font_size,
    legend_font_size=legend_font_size,
    # Ensure layout is determined before adding shapes/annotations relative to axes
    xaxis_range=[x_axis_min, x_axis_max] if x_axis_min < x_axis_max else None,
    yaxis_range=[y_axis_min, y_axis_max] if y_axis_min < y_axis_max else None
)

# Update axes base appearance (Grid, Line visibility/color)
# Note: Ranges are set in update_layout now to ensure they are fixed before adding shapes
fig.update_xaxes(
    showline=show_x_axis_line,
    linewidth=1,
    linecolor=axis_line_color,
    showgrid=show_grid,
    gridwidth=1,
    gridcolor=grid_color,
    zeroline=False, # Often good to disable default zeroline when adding custom threshold lines
    showticklabels=show_x_tick_labels,
)
fig.update_yaxes(
    showline=show_y_axis_line,
    linewidth=1,
    linecolor=axis_line_color,
    showgrid=show_grid,
    gridwidth=1,
    gridcolor=grid_color,
    zeroline=False,
    showticklabels=show_y_tick_labels
)

# Add Quadrant Lines and Labels ---
if enable_quadrants and x_axis_min < x_axis_max and y_axis_min < y_axis_max:
    # --- NEW: Add Quadrant Background Color Shapes ---
    # Draw shapes behind other elements like grid lines and data
    # Coordinates use data axes ('x', 'y') and axis limits/thresholds
    fig.add_shape(type="rect", layer="below", line_width=0,
                  xref="x", yref="y",
                  x0=x_threshold, y0=y_threshold, x1=x_axis_max, y1=y_axis_max,
                  fillcolor=color_top_right)
    fig.add_shape(type="rect", layer="below", line_width=0,
                  xref="x", yref="y",
                  x0=x_axis_min, y0=y_threshold, x1=x_threshold, y1=y_axis_max,
                  fillcolor=color_top_left)
    fig.add_shape(type="rect", layer="below", line_width=0,
                  xref="x", yref="y",
                  x0=x_axis_min, y0=y_axis_min, x1=x_threshold, y1=y_threshold,
                  fillcolor=color_bottom_left)
    fig.add_shape(type="rect", layer="below", line_width=0,
                  xref="x", yref="y",
                  x0=x_threshold, y0=y_axis_min, x1=x_axis_max, y1=y_threshold,
                  fillcolor=color_bottom_right)
    
    # Add Quadrant Lines
    fig.add_vline(
        x=x_threshold,
        line_width=quadrant_line_width,
        line_dash=quadrant_line_dash,
        line_color=quadrant_line_color,
    )
    fig.add_hline(
        y=y_threshold,
        line_width=quadrant_line_width,
        line_dash=quadrant_line_dash,
        line_color=quadrant_line_color,
    )

    # Define label positions (relative to current axis range)
    # Place labels slightly offset from the corners within each quadrant
    x_pad = (x_axis_max - x_axis_min) * 0.05 # 5% padding from edges/thresholds
    y_pad = (y_axis_max - y_axis_min) * 0.05

    # Top-Right Quadrant
    fig.add_annotation(
        x=x_axis_max - x_pad, y=y_axis_max - y_pad, xref="x", yref="y",
        text=label_top_right, showarrow=False, ax=0, ay=0, # No arrow, direct position
        font=dict(size=quadrant_label_font_size, color=quadrant_label_font_color),
        align="right", # Align text relative to anchor point
        xanchor='right', yanchor='top', # Anchor point on the text box itself
    )
    # Top-Left Quadrant
    fig.add_annotation(
        x=x_axis_min + x_pad, y=y_axis_max - y_pad, xref="x", yref="y",
        text=label_top_left, showarrow=False, ax=0, ay=0,
        font=dict(size=quadrant_label_font_size, color=quadrant_label_font_color),
        align="left",
        xanchor='left', yanchor='top',
    )
    # Bottom-Left Quadrant
    fig.add_annotation(
        x=x_axis_min + x_pad, y=y_axis_min + y_pad, xref="x", yref="y",
        text=label_bottom_left, showarrow=False, ax=0, ay=0,
        font=dict(size=quadrant_label_font_size, color=quadrant_label_font_color),
        align="left",
        xanchor='left', yanchor='bottom',
    )
    # Bottom-Right Quadrant
    fig.add_annotation(
        x=x_axis_max - x_pad, y=y_axis_min + y_pad, xref="x", yref="y",
        text=label_bottom_right, showarrow=False, ax=0, ay=0,
        font=dict(size=quadrant_label_font_size, color=quadrant_label_font_color),
        align="right",
        xanchor='right', yanchor='bottom',
    )

# Update point color and size
fig.update_traces(marker=dict(color=point_color, size=point_size))

# Update trace properties (point labels)
if show_point_labels:
    fig.update_traces(
        textfont_size=point_label_font_size,
        textfont_color=point_label_font_color,
        textposition='top right'
    )
else:
     fig.update_traces(text=None)

# --- 5. Display in Streamlit ---
st.plotly_chart(fig, use_container_width=True)
############################### CHART SECTION END ###########################
#############################################################################

st.divider()
with st.expander("View Raw Response"):
    st.dataframe(df)

# --- Footer/Instructions ---
st.markdown("---")
st.markdown("This web app was developed for demonstration purposes only.")
st.markdown("WIN Consulting")
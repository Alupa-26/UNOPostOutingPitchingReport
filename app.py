import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template
import io
import base64
import os

# --- 1. CORE FUNCTIONS ---

pitch_type_colors = {
    "FB": "tomato",
    "2S": "orange",
    "CT": "gold",
    "SL": "deepskyblue",
    "CB": "aqua",
    "CH": "limegreen",
    "SP": "olivedrab",
    "SW": "Lime",
    "UN": "gray",
    "OS": "gray"
}

def mavsplus_color_gradient(val):
    if not isinstance(val, (int, float)) or pd.isna(val):
        return 'background-color: #FFFFFF'
    
    val = max(0, min(val, 151))
    
    colors = [
        (0, '#FF0000'),   
        (100, '#FFFFFF'), 
        (151, '#00FF00')  
    ]
    
    for i in range(len(colors) - 1):
        val1, color1 = colors[i]
        val2, color2 = colors[i + 1]
        if val1 <= val <= val2:
            ratio = (val - val1) / (val2 - val1)
            r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
            r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            return f'background-color: {color}'
    
    return 'background-color: #FFFFFF'

def map_pitch_type(tagged_pitch_type):
    """Map raw pitch types to their labels."""
    if pd.isna(tagged_pitch_type):
        return "UN"
        
    pitch_mapping = {
        "FourSeamFastBall": "FB", "Fastball": "FB",
        "TwoSeamFastBall": "2S", "Sinker": "2S",
        "Cutter": "CT",
        "Slider": "SL",
        "Curveball": "CB",
        "ChangeUp": "CH",
        "Splitter": "SP",
        "Sweeper": "SW",
        "Undefined": "UN", "Other": "OS"
    }
    return pitch_mapping.get(str(tagged_pitch_type), str(tagged_pitch_type)[:2].upper())

def calculate_splits(data, batter_side=None):
    """Calculate splits for Total, Vs. RHH, Vs. LHH."""
    if batter_side:
        filtered_data = data[data['BatterSide'] == batter_side]
    else:
        filtered_data = data

    total_pitches = len(filtered_data)
    batters_faced = len(filtered_data[filtered_data['PitchofPA'] == 1])
    
    if total_pitches == 0:
        return {
            "Splits": "Total" if batter_side is None else ("Vs. RHH" if batter_side == "Right" else "Vs. LHH"),
            "H": 0, "XBH": 0, "Hard": 0, "Barrel": 0, "SO": 0, "BB": 0, "HBP": 0,
            "Strike": "0% (0/0)", "KZone": "0% (0/0)", "First": "0% (0/0)",
            "FirstThree": "0% (0/0)", "CSW": "0% (0/0)", "Chase": "0% (0/0)",
            "BF": 0, "Pitches": 0
        }

    chase_pitches = len(filtered_data[
        ~((filtered_data['PlateLocSide'].between(-0.8, 0.8)) & 
          (filtered_data['PlateLocHeight'].between(1.575, 3.575))) & 
        (filtered_data['PitchCall'].isin(['InPlay', 'StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable']))
    ])
    chase_total = filtered_data['PitchCall'].isin(['InPlay', 'StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable']).sum()

    stats = {
        "Splits": "Total" if batter_side is None else ("Vs. RHH" if batter_side == "Right" else "Vs. LHH"),
        "H": filtered_data['PlayResult'].isin(["Single", "Double", "Triple", "HomeRun"]).sum(),
        "XBH": filtered_data['PlayResult'].isin(["Double", "Triple", "HomeRun"]).sum(),
        "Hard": len(filtered_data[(filtered_data['PitchCall'] == "InPlay") & (filtered_data['ExitSpeed'] > 96)]),
        "Barrel": len(filtered_data[(filtered_data['PlayResult'].isin(["Single", "Double", "Triple", "HomeRun"])) &
                                     (filtered_data['ExitSpeed'] > 96) &
                                     (filtered_data['Angle'].between(15, 25))]),
        "SO": filtered_data['KorBB'].eq("Strikeout").sum(),
        "BB": filtered_data['KorBB'].eq("Walk").sum(),
        "HBP": filtered_data['PitchCall'].eq("HitByPitch").sum(),
        "Strike": f"{filtered_data['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay']).sum() / total_pitches:.0%} ({filtered_data['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay']).sum()}/{total_pitches})",
        "KZone": f"{len(filtered_data[(filtered_data['PlateLocSide'].between(-0.8, 0.8)) & (filtered_data['PlateLocHeight'].between(1.575, 3.575))]) / total_pitches:.0%} ({len(filtered_data[(filtered_data['PlateLocSide'].between(-0.8, 0.8)) & (filtered_data['PlateLocHeight'].between(1.575, 3.575))])}/{total_pitches})",
        "First": f"{len(filtered_data[(filtered_data['PitchofPA'] == 1) & (filtered_data['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable']))]) / batters_faced:.0%} ({len(filtered_data[(filtered_data['PitchofPA'] == 1) & (filtered_data['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable']))])}/{batters_faced})" if batters_faced > 0 else "0% (0/0)",
        "FirstThree": f"{len(filtered_data[(filtered_data['PitchofPA'] <= 3) & ((filtered_data['KorBB'] == 'Strikeout') | (filtered_data['PlayResult'] == 'Out'))]) / batters_faced:.0%} ({len(filtered_data[(filtered_data['PitchofPA'] <= 3) & ((filtered_data['KorBB'] == 'Strikeout') | (filtered_data['PlayResult'] == 'Out'))])}/{batters_faced})" if batters_faced > 0 else "0% (0/0)",
        "CSW": f"{filtered_data['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging']).sum() / total_pitches:.0%} ({filtered_data['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging']).sum()}/{total_pitches})",
        "Chase": f"{chase_pitches / chase_total:.0%} ({chase_pitches}/{chase_total})" if chase_total > 0 else "0% (0/0)",
        "BF": batters_faced,
        "Pitches": total_pitches
    }

    return stats

def calculate_pitch_metrics(data):
    """Calculate metrics for each pitch type."""
    data['PitchTypeMapped'] = data['TaggedPitchType'].map(map_pitch_type)
    pitch_groups = data.groupby('PitchTypeMapped')

    metrics = []

    for pitch_type, group in pitch_groups:
        total_pitches = len(group)
        strikes = group['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable']).sum()
        in_kzone = len(group[(group['PlateLocSide'].between(-0.8, 0.8)) & (group['PlateLocHeight'].between(1.575, 3.575))])
        chase_pitches = len(group[~((group['PlateLocSide'].between(-0.8, 0.8)) & (group['PlateLocHeight'].between(1.575, 3.575))) & group['PitchCall'].isin(['StrikeSwinging', 'InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable'])])
        pitches_in_chase = group['PitchCall'].isin(['StrikeSwinging', 'InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable']).sum()

        metrics.append({
            "PitchType": pitch_type,
            "Velocity": f"{int(group['RelSpeed'].min())}-{int(group['RelSpeed'].mean())}, T{int(group['RelSpeed'].max())}",
            "SpinRate": f"{int(group['SpinRate'].min())}-{int(group['SpinRate'].max())}",
            "IVB": f"{group['InducedVertBreak'].mean():.1f}",
            "HB": f"{group['HorzBreak'].mean():.1f}",
            "VAA": f"{group['VertApprAngle'].mean():.1f}",
            "VRA": f"{group['VertRelAngle'].mean():.1f}",
            "Extension": f"{group['Extension'].mean():.1f}",
            "Height": f"{group['RelHeight'].mean():.1f}",
            "Side": f"{group['RelSide'].mean():.1f}",
            "Strike": f"{strikes / total_pitches:.0%} ({strikes}/{total_pitches})",
            "KZone": f"{in_kzone / total_pitches:.0%} ({in_kzone}/{total_pitches})",
            "CSW": f"{strikes / total_pitches:.0%} ({strikes}/{total_pitches})",
            "Chase": f"{chase_pitches / pitches_in_chase:.0%} ({chase_pitches}/{pitches_in_chase})" if pitches_in_chase > 0 else "0% (0/0)",
            "Pitches": total_pitches,
            "MAVSPLUS": round(group['MAVSPLUS'].mean(), 1)
        })

    # Sort by most thrown pitch type
    metrics.sort(key=lambda x: x['Pitches'], reverse=True)
    return pd.DataFrame(metrics)

def create_plot(data, title, condition):
    """Create a plot with KZone and Shadow Zone."""
    plt.figure(figsize=(3, 5))  
    plt.title(title, fontsize=16)  
    plt.xlabel("Plate Loc Side")
    plt.ylabel("Plate Loc Height")

    # KZone and Shadow Zone
    kzone_x = [0.8, 0.8, -0.8, -0.8, 0.8]
    kzone_y = [3.575, 1.575, 1.575, 3.575, 3.575]
    shadow_zone_x = [1, 1, -1, -1, 1]
    shadow_zone_y = [3.775, 1.375, 1.375, 3.775, 3.775]

    plt.fill(shadow_zone_x, shadow_zone_y, color="lightgray", alpha=0.5)
    plt.plot(kzone_x, kzone_y, color="black", linewidth=2)

    # Add black lines at x=0.0 and y=2.5
    plt.axvline(x=0.0, color="black", linewidth=1.5, linestyle="-")  
    plt.axhline(y=2.5, color="black", linewidth=1.5, linestyle="-")  

    filtered_data = data[condition]
    for _, row in filtered_data.iterrows():
        pitch_color = pitch_type_colors.get(map_pitch_type(row['TaggedPitchType']), "gray")
        plt.scatter(row['PlateLocSide'], row['PlateLocHeight'], color=pitch_color, edgecolor="black", s=60)

    # Update axis ranges
    plt.xlim(-1.5, 1.5)  
    plt.ylim(0.5, 4.5)  

    # Set ticks with 0.5 increments for X-axis
    plt.xticks(ticks=[x * 0.5 for x in range(-3, 4)])  
    plt.yticks(ticks=[y * 0.5 for y in range(1, 10)])  

    # Remove internal grid lines
    plt.grid(False)

    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="margin:10px; width:22%; display:inline-block; vertical-align:top;"/>'

def generate_plots(data):
    """Generate all five plots."""
    plots = []
    plots.append(create_plot(data, "First", condition=data['PitchofPA'] == 1))
    plots.append(create_plot(data, "2-Strike", condition=data['Strikes'] == 2))
    plots.append(create_plot(data, "Damage",
                             condition=(data['PlayResult'].isin(["Double", "Triple", "HomeRun"]) |
                                        ((data['ExitSpeed'] > 96) & (data['Angle'].between(15, 25))))))
    plots.append(create_plot(data, "CSW", condition=data['PitchCall'].isin(["StrikeCalled", "StrikeSwinging"])))
    plots.append(create_plot(data, "Full Location", condition=data.index.notnull()))
    return f'<div style="display: flex; justify-content: flex-start; align-items: flex-start;">{"".join(plots)}</div>'

def calculate_at_bat_overview(data):
    """Create the At-Bat Overview table based on 'PitchofPA' resets."""
    at_bats = []
    grouped_data = data.groupby((data['PitchofPA'] == 1).cumsum())  

    for _, group in grouped_data:
        first_row = group.iloc[0]  
        last_row = group.iloc[-1]  

        # Abbreviations for PlayResult, KorBB, and PitchCall
        play_result_map = {
            "Single": "1B", "Double": "2B", "Triple": "3B", "HomeRun": "HR",
            "FieldersChoice": "FC", "Out": "Out", "Sacrifice": "Sac", "Error": "E"
        }
        korbb_map = {"Strikeout": "SO", "Walk": "BB"}
        pitch_call_map = {
            "HitByPitch": "HBP", "StrikeSwinging": "S", "StrikeCalled": "K",
            "BallCalled": "B", "FoulBallFieldable": "F", "FoulBallNotFieldable": "F",
            "InPlay": "IP"
        }
        pitch_type_map = {
            "FourSeamFastBall": "FB", "Fastball": "FB", "TwoSeamFastBall": "SI",
            "Sinker": "SI", "Cutter": "CT", "Slider": "SL", "Curveball": "CU",
            "ChangeUp": "CH", "Splitter": "SP"
        }
        tagged_hit_type_map = {
            "FlyBall": "FB", "GroundBall": "GB", "LineDrive": "LD",
            "Popup": "PU", "Bunt": "B", "Undefined": "-"
        }

        # Determine Result
        result = (
            play_result_map.get(last_row['PlayResult'], "")
            or korbb_map.get(last_row['KorBB'], "")
            or pitch_call_map.get(last_row['PitchCall'], "")
        )

        # Pitch Sequence
        sequence = "".join(
            pitch_call_map.get(pitch, pitch[:1])  
            for pitch in group['PitchCall']
        )

        # Tagged Pitch Type
        tagged_pitch_type = pitch_type_map.get(last_row['TaggedPitchType'], "-")

        # BIP (TaggedHitType)
        bip = tagged_hit_type_map.get(last_row['TaggedHitType'], "-")

        # Exit Velocity (EV) and Launch Angle (LA)
        ev = int(last_row['ExitSpeed']) if pd.notna(last_row['ExitSpeed']) else "-"
        la = int(last_row['Angle']) if pd.notna(last_row['Angle']) else "-"

        # Add At-Bat row
        at_bats.append({
            "Batter": first_row['Batter'],  
            "Result": result,
            "Seq": sequence,
            "TaggedPitchType": tagged_pitch_type,
            "BIP": bip,
            "EV": ev,
            "LA": la
        })

    return pd.DataFrame(at_bats)

def create_movement_plot(data, pitcher_throws):
    """Generate Movement Plot with consistent sizing."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)  

    for _, row in data.iterrows():
        pitch_color = pitch_type_colors.get(map_pitch_type(row['TaggedPitchType']), "gray")
        ax.scatter(row['HorzBreak'], row['InducedVertBreak'], color=pitch_color, s=40, edgecolor="black")

    # Add bold axis lines
    ax.axhline(0, color="black", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1.5)

    # Format axis limits and ticks
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xticks(range(-25, 30, 5))
    ax.set_yticks(range(-25, 30, 5))
    ax.set_aspect('equal', adjustable='box')  

    # Adjust tick size
    ax.tick_params(axis='both', which='major', labelsize=6)  
    ax.tick_params(axis='both', which='minor', labelsize=6)  

    # Add axis labels based on handedness
    if pitcher_throws == "Right":
        ax.text(23, 0, "Arm Side", fontsize=8, color="white", ha="left", va="center", weight="bold",
                bbox=dict(facecolor="grey", edgecolor="none", pad=2))
        ax.text(-23, 0, "Glove Side", fontsize=8, color="white", ha="right", va="center", weight="bold",
                bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    else:  # Left-handed
        ax.text(-23, 0, "Arm Side", fontsize=8, color="white", ha="right", va="center", weight="bold",
                bbox=dict(facecolor="grey", edgecolor="none", pad=2))
        ax.text(23, 0, "Glove Side", fontsize=8, color="white", ha="left", va="center", weight="bold",
                bbox=dict(facecolor="grey", edgecolor="none", pad=2))

    ax.text(0, 23, "Carry", fontsize=8, color="white", ha="center", va="bottom", weight="bold",
            bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    ax.text(0, -23, "Sink", fontsize=8, color="white", ha="center", va="top", weight="bold",
            bbox=dict(facecolor="grey", edgecolor="none", pad=2))

    # Add Title
    ax.set_title("Movement Plot", fontsize=12, weight="bold")

    # Save as base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")  
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="width: 100%; height: auto; display: block;"/>'
    
def create_release_plot(data):
    """Generate a smaller Release Plot with adjusted tick sizes."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)  

    for _, row in data.iterrows():
        pitch_color = pitch_type_colors.get(map_pitch_type(row['TaggedPitchType']), "gray")
        ax.scatter(row['RelSide'], row['RelHeight'], color=pitch_color, s=40, edgecolor="black")

    # Add bold axis line
    ax.axvline(0, color="black", linewidth=1)

    # Format axis limits and ticks
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 7)
    ax.set_xticks(range(-4, 5, 1))
    ax.set_yticks(range(0, 8, 1))
    ax.set_aspect('equal', adjustable='box')  

    # Adjust tick and number font sizes
    ax.tick_params(axis='both', which='major', labelsize=6)  
    ax.tick_params(axis='both', which='minor', labelsize=6)  

    # Add Title
    ax.set_title("Release Plot", fontsize=12, weight="bold")

    # Save as base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")  
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="width: 80%; height: auto; display: block;"/>'

def create_vertical_stacked_bar_chart(data, condition, title):
    """Create a vertical stacked bar chart for pitch usage."""
    filtered_data = data[condition]

    # Calculate percentages
    pitch_counts = filtered_data['TaggedPitchType'].map(map_pitch_type).value_counts()
    total_pitches = pitch_counts.sum()
    pitch_percentages = (pitch_counts / total_pitches * 100).round(1)

    # Setup for stacked bar chart
    fig, ax = plt.subplots(figsize=(1.5, 4))  
    pitch_labels = pitch_percentages.index
    
    pitch_colors = [pitch_type_colors.get(pt, "gray") for pt in pitch_labels]

    # Cumulative height for stacking
    cumulative_height = 0

    for pitch_type, pct, color in zip(pitch_labels, pitch_percentages, pitch_colors):
        # Plot the bar segment
        bar = ax.bar(
            x=0,  
            height=pct,
            width=0.4,
            color=color,
            edgecolor="black",
            bottom=cumulative_height,
            label=pitch_type
        )
        # Add text inside the bar
        ax.text(
            0,  
            cumulative_height + pct / 2,  
            f"{pitch_type} {pct}%",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            weight="bold"
        )
        # Update cumulative height
        cumulative_height += pct

    # Formatting
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 0)  
    ax.set_xticks([])
    ax.set_title(title, fontsize=12, weight="bold")
    plt.tight_layout()

    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="width: auto; height: auto; display: block;"/>'

def generate_pitch_usage_charts(data):
    """Generate all five pitch usage stacked bar charts."""
    charts = []

    # First
    charts.append(create_vertical_stacked_bar_chart(
        data,
        condition=(data['PitchofPA'] == 1),
        title="First"
    ))

    # Even
    charts.append(create_vertical_stacked_bar_chart(
        data,
        condition=(data['Balls'] == data['Strikes']),
        title="Even"
    ))

    # Hitter
    charts.append(create_vertical_stacked_bar_chart(
        data,
        condition=((data['Balls'] == 3) & (data['Strikes'] <= 1)) | ((data['Balls'] == 2) & (data['Strikes'] == 0)),
        title="Hitter"
    ))

    # 2-Strikes
    charts.append(create_vertical_stacked_bar_chart(
        data,
        condition=(data['Strikes'] == 2),
        title="2-Strikes"
    ))

    # Overall
    charts.append(create_vertical_stacked_bar_chart(
        data,
        condition=data.index.notnull(),
        title="Overall"
    ))

    return f'<div style="display: flex; flex-wrap: wrap; justify-content: center;">{"".join(charts)}</div>'

# Ensure the logo embeds correctly into the HTML by using base64
def get_base64_logo(file_path):
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        return "" # Returns empty if no logo is found

# --- 2. STREAMLIT UI ---

# Dynamic Pathing: Allows logo to be found locally or on the Streamlit cloud server automatically.
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, 'logo.png')
logo_base64 = get_base64_logo(logo_path) 

# Clean Page Configuration
st.set_page_config(page_title="Nebraska Omaha Post Outing Report", page_icon="⚾", layout="wide")

# Inject Custom CSS for a highly professional look
custom_css = """
<style>
/* Hide default Streamlit top menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom File Uploader styling */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #b3b3b3 !important;
    border-radius: 12px;
    background-color: #fafafa;
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: #f0f0f0;
    border-color: #cc0000 !important;
}

/* Fancy Download Button styling */
.stDownloadButton button {
    background: linear-gradient(135deg, #e3000f 0%, #a60000 100%);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 12px 28px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    width: auto;
}
.stDownloadButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    background: linear-gradient(135deg, #ff1a2b 0%, #cc0000 100%);
    color: white !important;
    border-color: transparent;
}
.stDownloadButton button:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px white, 0 0 0 4px #e3000f !important;
    color: white !important;
}
.stDownloadButton button p {
    color: white !important;
    font-size: 18px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Build the professional header
col1, col2 = st.columns([1, 6])
with col1:
    # Fix: Explicitly check if the image file exists on the server before trying to render it
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
with col2:
    st.markdown("<h1 style='text-align: left; margin-top: -15px; color: #111;'>Nebraska Omaha Post Outing Report</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px; color: gray;'>Upload a Trackman CSV file below to generate the formatted HTML report.</p>", unsafe_allow_html=True)

st.divider()

# File uploader
uploaded_file = st.file_uploader("Select Game CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Clean column names
    data.columns = data.columns.str.strip().str.replace('\ufeff', '')
    
    # Ensure Cougs+ is available as 'MAVSPLUS'
    if 'MAVSPLUS' in data.columns:
        data['MAVSPLUS'] = data['MAVSPLUS']
    else:
        data['MAVSPLUS'] = None  # fallback if column is missing

    st.success("File successfully loaded! Generating report...")
    
    with st.spinner("Crunching data and generating plots..."):
        html_sections = []
        unique_pitchers = data['Pitcher'].unique()

        for pitcher in unique_pitchers:
            pitcher_data = data[data['Pitcher'] == pitcher]

            if pitcher_data.empty:
                continue

            report_date = pitcher_data['Date'].iloc[0]
            pitcher_throws = pitcher_data['PitcherThrows'].iloc[0]  

            # Split Table
            split_rows = [
                calculate_splits(pitcher_data),
                calculate_splits(pitcher_data, batter_side="Right"),
                calculate_splits(pitcher_data, batter_side="Left")
            ]
            df_splits = pd.DataFrame(split_rows)

            # Pitch Metric Table
            df_metrics = calculate_pitch_metrics(pitcher_data)

            # Apply conditional formatting to MAVSPLUS and row colors
            def apply_row_colors(row):
                pitch_type = row['PitchType']
                color = pitch_type_colors.get(pitch_type, 'white')
                return [f'background-color: {color}' if col != 'MAVSPLUS' else '' for col in df_metrics.columns]

            # Use map instead of deprecated applymap for newer pandas versions
            df_metrics_styled = df_metrics.style.map(
                mavsplus_color_gradient, subset=['MAVSPLUS']
            ).apply(
                apply_row_colors, axis=1
            ).format(
                {'MAVSPLUS': '{:.1f}'}
            ).to_html(
                classes='pitch-metric-table',
                index=False,
                escape=False
            )

            # At-Bat Overview Table
            df_at_bat_overview = calculate_at_bat_overview(pitcher_data)

            # Generate Plots
            plots_html = generate_plots(pitcher_data)

            # Generate Movement Plot
            movement_plot_html = create_movement_plot(pitcher_data, pitcher_throws)

            # Generate Release Plot
            release_plot_html = create_release_plot(pitcher_data)

            # Generate Pitch Usage Charts
            pitch_usage_html = generate_pitch_usage_charts(pitcher_data)

            # HTML Section
            template = Template("""
<div style="margin-bottom: 50px; padding: 30px; border: 4px solid #333; border-radius: 8px; position: relative; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <div style="position: absolute; top: 20px; left: 20px; z-index: 2;">
        <img src="{{ logo_base64 }}" alt="Logo" style="width: 400px; height: auto;">
    </div>

    <div style="text-align: center; margin-top: 40px;">
        <h2 style="color: red; margin-bottom: 20px; background: rgba(255, 255, 255, 0.8); display: inline-block; padding: 10px;">
            {{ date }}, {{ pitcher_name }} Post Outing Report
        </h2>
    </div>

    <table style="width:100%; margin-bottom:20px; border-collapse:collapse; text-align:center;">
        <thead>
            <tr>
                {% for col in df_splits.columns %}
                <th style="border: 1px solid #ddd; padding: 8px; background-color: #f4f4f4;">{{ col }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in df_splits.values %}
            <tr>
                {% for cell in row %}
                <td style="border: 1px solid #ddd; padding: 8px;">{{ cell }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>

    {{ df_metrics_styled|safe }}

    <div style="display: flex; flex-direction: row; justify-content: space-between; gap: 20px; margin-top: 20px;">
        <div style="width: 65%;">
            {{ plots_html }}
        </div>

        <div style="width: 30%;">
            <h3 style="text-align: center; margin-bottom: 10px;">At-Bat Overview</h3>
            <table style="width: 100%; border-collapse: collapse; text-align: center;">
                <thead>
                    <tr>
                        {% for col in df_at_bat_overview.columns %}
                        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f4f4f4;">
                            {{ "Pitch" if col == "TaggedPitchType" else col }}
                        </th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for row in df_at_bat_overview.values %}
                    <tr>
                        {% for cell in row %}
                        <td style="border: 1px solid #ddd; padding: 8px;">{{ cell }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: flex-start; gap: 30px; margin-top: 50px;">
        <div style="flex: 1; height: 600px; display: flex; align-items: center; justify-content: center;">
            {{ movement_plot_html }}
        </div>

        <div style="flex: 1; height: 600px; display: flex; align-items: center; justify-content: center;">
            {{ release_plot_html }}
        </div>

        <div style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <h3 style="text-align: center; margin-bottom: 10px; font-size: 45px;">Pitch Usage</h3>
            <div style="display: flex; flex-direction: row; justify-content: space-around; align-items: center; gap: 0px; width: 100%; height: 100%;">
                {{ pitch_usage_html }}
            </div>
        </div>
    </div>
</div>
""")
            # Render HTML Section
            html_sections.append(template.render(
                pitcher_name=pitcher,
                date=report_date,
                df_splits=df_splits,
                df_metrics=df_metrics,
                df_metrics_styled=df_metrics_styled,
                df_at_bat_overview=df_at_bat_overview,
                plots_html=plots_html,
                movement_plot_html=movement_plot_html,
                release_plot_html=release_plot_html,
                pitch_type_colors=pitch_type_colors,
                logo_base64=logo_base64,
                pitch_usage_html=pitch_usage_html
            ))

        # Final HTML Template
        final_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Post Outing Reports</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h2 { font-size: 75px; color: red; }
        table { margin: 20px 0; width: 100%; border-collapse: collapse; }
        th, td { text-align: center; padding: 8px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1 style="text-align:center; color: red;">Post Outing Reports</h1>
    {% for section in sections %}
        {{ section|safe }}
    {% endfor %}
</body>
</html>
""")

        # Render Final HTML
        html_content = final_template.render(sections=html_sections)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Styled center column for the single download button
    col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
    with col_center:
        st.download_button(
            label="📄 Download Full HTML Report",
            data=html_content,
            file_name="pitching_report.html",
            mime="text/html",
            use_container_width=True
        )
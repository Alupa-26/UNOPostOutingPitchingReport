import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from jinja2 import Template
import io
import base64
import os

# --- 1. GLOBAL VARIABLES & SHARED FUNCTIONS ---

pitch_type_colors = {
    "FB": "tomato",
    "2S": "orange",
    "CT": "gold",
    "SL": "deepskyblue",
    "CB": "aqua",
    "CH": "limegreen",
    "SP": "olivedrab",
    "SW": "Lime",
    "KN": "pink",
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
        "Knuckleball": "KN",
        "Undefined": "UN", "Other": "OS"
    }
    return pitch_mapping.get(str(tagged_pitch_type), str(tagged_pitch_type)[:2].upper())


# --- 2. POST-OUTING SPECIFIC FUNCTIONS ---

def calculate_splits(data, batter_side=None):
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

def calculate_pitch_metrics_post_outing(data):
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
    metrics.sort(key=lambda x: x['Pitches'], reverse=True)
    return pd.DataFrame(metrics)

def create_plot(data, title, condition):
    fig, ax = plt.subplots(figsize=(3, 5))  
    ax.set_title(title, fontsize=16)  
    ax.set_xlabel("Plate Loc Side")
    ax.set_ylabel("Plate Loc Height")
    
    kzone_x = [0.8, 0.8, -0.8, -0.8, 0.8]
    kzone_y = [3.575, 1.575, 1.575, 3.575, 3.575]
    shadow_zone_x = [1, 1, -1, -1, 1]
    shadow_zone_y = [3.775, 1.375, 1.375, 3.775, 3.775]
    
    ax.fill(shadow_zone_x, shadow_zone_y, color="lightgray", alpha=0.5)
    ax.plot(kzone_x, kzone_y, color="black", linewidth=2)
    ax.axvline(x=0.0, color="black", linewidth=1.5, linestyle="-")  
    ax.axhline(y=2.5, color="black", linewidth=1.5, linestyle="-")  

    filtered_data = data[condition]
    for _, row in filtered_data.iterrows():
        pitch_color = pitch_type_colors.get(map_pitch_type(row['TaggedPitchType']), "gray")
        ax.scatter(row['PlateLocSide'], row['PlateLocHeight'], color=pitch_color, edgecolor="black", s=60)

    ax.set_xlim(-1.5, 1.5)  
    ax.set_ylim(0.5, 4.5)  
    ax.set_xticks(ticks=[x * 0.5 for x in range(-3, 4)])  
    ax.set_yticks(ticks=[y * 0.5 for y in range(1, 10)])  
    ax.grid(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="margin:10px; width:22%; display:inline-block; vertical-align:top;"/>'

def generate_plots(data):
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
    at_bats = []
    grouped_data = data.groupby((data['PitchofPA'] == 1).cumsum())  
    for _, group in grouped_data:
        first_row = group.iloc[0]  
        last_row = group.iloc[-1]  
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
        result = (
            play_result_map.get(last_row['PlayResult'], "")
            or korbb_map.get(last_row['KorBB'], "")
            or pitch_call_map.get(last_row['PitchCall'], "")
        )
        sequence = "".join(
            pitch_call_map.get(pitch, pitch[:1])  
            for pitch in group['PitchCall']
        )
        tagged_pitch_type = pitch_type_map.get(last_row['TaggedPitchType'], "-")
        bip = tagged_hit_type_map.get(last_row['TaggedHitType'], "-")
        ev = int(last_row['ExitSpeed']) if pd.notna(last_row['ExitSpeed']) else "-"
        la = int(last_row['Angle']) if pd.notna(last_row['Angle']) else "-"
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

def create_movement_plot_post_outing(data, pitcher_throws):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)  
    for _, row in data.iterrows():
        pitch_color = pitch_type_colors.get(map_pitch_type(row['TaggedPitchType']), "gray")
        ax.scatter(row['HorzBreak'], row['InducedVertBreak'], color=pitch_color, s=40, edgecolor="black")
    
    ax.axhline(0, color="black", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xticks(range(-25, 30, 5))
    ax.set_yticks(range(-25, 30, 5))
    ax.set_aspect('equal', adjustable='box')  
    ax.tick_params(axis='both', which='major', labelsize=6)  
    ax.tick_params(axis='both', which='minor', labelsize=6)  
    
    if pitcher_throws == "Right":
        ax.text(23, 0, "Arm Side", fontsize=8, color="white", ha="left", va="center", weight="bold",
                bbox=dict(facecolor="grey", edgecolor="none", pad=2))
        ax.text(-23, 0, "Glove Side", fontsize=8, color="white", ha="right", va="center", weight="bold",
                bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    else:  
        ax.text(-23, 0, "Arm Side", fontsize=8, color="white", ha="right", va="center", weight="bold",
                bbox=dict(facecolor="grey", edgecolor="none", pad=2))
        ax.text(23, 0, "Glove Side", fontsize=8, color="white", ha="left", va="center", weight="bold",
                bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    ax.text(0, 23, "Carry", fontsize=8, color="white", ha="center", va="bottom", weight="bold",
            bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    ax.text(0, -23, "Sink", fontsize=8, color="white", ha="center", va="top", weight="bold",
            bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    ax.set_title("Movement Plot", fontsize=12, weight="bold")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")  
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="width: 100%; height: auto; display: block;"/>'
    
def create_release_plot_post_outing(data):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)  
    for _, row in data.iterrows():
        pitch_color = pitch_type_colors.get(map_pitch_type(row['TaggedPitchType']), "gray")
        ax.scatter(row['RelSide'], row['RelHeight'], color=pitch_color, s=40, edgecolor="black")
    
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 7)
    ax.set_xticks(range(-4, 5, 1))
    ax.set_yticks(range(0, 8, 1))
    ax.set_aspect('equal', adjustable='box')  
    ax.tick_params(axis='both', which='major', labelsize=6)  
    ax.tick_params(axis='both', which='minor', labelsize=6)  
    ax.set_title("Release Plot", fontsize=12, weight="bold")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")  
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="width: 80%; height: auto; display: block;"/>'

def create_vertical_stacked_bar_chart(data, condition, title):
    filtered_data = data[condition]
    pitch_counts = filtered_data['TaggedPitchType'].map(map_pitch_type).value_counts()
    total_pitches = pitch_counts.sum()
    pitch_percentages = (pitch_counts / total_pitches * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(1.5, 4))  
    pitch_labels = pitch_percentages.index
    pitch_colors = [pitch_type_colors.get(pt, "gray") for pt in pitch_labels]
    
    cumulative_height = 0
    for pitch_type, pct, color in zip(pitch_labels, pitch_percentages, pitch_colors):
        ax.bar(x=0, height=pct, width=0.4, color=color, edgecolor="black", bottom=cumulative_height, label=pitch_type)
        ax.text(0, cumulative_height + pct / 2, f"{pitch_type} {pct}%", ha="center", va="center", fontsize=12, color="black", weight="bold")
        cumulative_height += pct
        
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 0.5)  
    ax.set_xticks([])
    ax.set_title(title, fontsize=12, weight="bold")
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="width: auto; height: auto; display: block;"/>'

def generate_pitch_usage_charts(data):
    charts = []
    charts.append(create_vertical_stacked_bar_chart(data, condition=(data['PitchofPA'] == 1), title="First"))
    charts.append(create_vertical_stacked_bar_chart(data, condition=(data['Balls'] == data['Strikes']), title="Even"))
    charts.append(create_vertical_stacked_bar_chart(data, condition=((data['Balls'] == 3) & (data['Strikes'] <= 1)) | ((data['Balls'] == 2) & (data['Strikes'] == 0)), title="Hitter"))
    charts.append(create_vertical_stacked_bar_chart(data, condition=(data['Strikes'] == 2), title="2-Strikes"))
    charts.append(create_vertical_stacked_bar_chart(data, condition=data.index.notnull(), title="Overall"))
    return f'<div style="display: flex; flex-wrap: wrap; justify-content: center;">{"".join(charts)}</div>'


# --- 3. BULLPEN SPECIFIC FUNCTIONS ---

def calculate_pitch_metrics_bullpen(data):
    data['PitchTypeMapped'] = data['TaggedPitchType'].map(map_pitch_type)
    pitch_groups = data.groupby('PitchTypeMapped')
    metrics = []

    for pitch_type, group in pitch_groups:
        total_pitches = len(group)
        strikes = group['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable']).sum()
        in_kzone = len(group[(group['PlateLocSide'].between(-0.8, 0.8)) & (group['PlateLocHeight'].between(1.575, 3.575))])
        
        mavsplus = group['MAVSPLUS'].mean()
        mavsplus_value = "NA" if pd.isna(mavsplus) else round(mavsplus, 1)

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
            "Pitches": total_pitches,
            "MAVSPLUS": mavsplus_value
        })

    metrics.sort(key=lambda x: x['Pitches'], reverse=True)
    return pd.DataFrame(metrics)

def create_location_plot(data):
    fig, ax = plt.subplots(figsize=(3, 4), dpi=200)  
    ax.set_title("Full Location", fontsize=12, weight="bold")
    ax.set_xlabel("Plate Loc Side", fontsize=8)
    ax.set_ylabel("Plate Loc Height", fontsize=8)
    kzone_x = [0.8, 0.8, -0.8, -0.8, 0.8]
    kzone_y = [3.575, 1.575, 1.575, 3.575, 3.575]
    shadow_zone_x = [1, 1, -1, -1, 1]
    shadow_zone_y = [3.775, 1.375, 1.375, 3.775, 3.775]
    ax.fill(shadow_zone_x, shadow_zone_y, color="lightgray", alpha=0.5)
    ax.plot(kzone_x, kzone_y, color="black", linewidth=1.5)
    ax.axvline(x=0.0, color="black", linewidth=1, linestyle="-")
    ax.axhline(y=2.5, color="black", linewidth=1, linestyle="-")

    for _, row in data.iterrows():
        pitch_color = pitch_type_colors.get(map_pitch_type(row['TaggedPitchType']), "gray")
        ax.scatter(row['PlateLocSide'], row['PlateLocHeight'], color=pitch_color, edgecolor="black", s=40)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0.5, 4.5)
    ax.set_xticks([x * 0.5 for x in range(-3, 4)])
    ax.set_yticks([y * 0.5 for y in range(1, 10)])
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="width:100%; height:100%; object-fit:contain; display:block;"/>'

def create_vaa_plot(data_group, title):
    fig, ax = plt.subplots(figsize=(3, 4), dpi=200)  
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Plate Loc Side', fontsize=8)
    ax.set_ylabel('Plate Loc Height', fontsize=8)
    strike_low = 1.575
    strike_high = 3.575
    mid = 2.5
    shadow_low = 1.375
    shadow_high = 3.775
    strike_horz = 0.8
    shadow_x = [1, 1, -1, -1, 1]
    shadow_y = [shadow_high, shadow_low, shadow_low, shadow_high, shadow_high]
    ax.fill(shadow_x, shadow_y, color='lightgray', alpha=0.5)
    kzone_x = [strike_horz, strike_horz, -strike_horz, -strike_horz, strike_horz]
    kzone_y = [strike_high, strike_low, strike_low, strike_high, strike_high]
    ax.plot(kzone_x, kzone_y, color='black', linewidth=1.5)
    ax.axvline(x=0.0, color='black', linewidth=1, linestyle='-')
    ax.axhline(y=mid, color='black', linewidth=1, linestyle='-')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0.5, 4.5)
    ax.set_xticks([x * 0.5 for x in range(-3, 4)])
    ax.set_yticks([y * 0.5 for y in range(1, 10)])
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(False)

    if not data_group.empty:
        in_column = data_group[abs(data_group['PlateLocSide']) <= strike_horz]
        above_df = in_column[in_column['PlateLocHeight'] > strike_high]
        below_df = in_column[in_column['PlateLocHeight'] < strike_low]
        upper_df = in_column[(in_column['PlateLocHeight'] > mid) & (in_column['PlateLocHeight'] <= strike_high)]
        lower_df = in_column[(in_column['PlateLocHeight'] <= mid) & (in_column['PlateLocHeight'] >= strike_low)]
        above_vaa = above_df['VertApprAngle'].mean()
        below_vaa = below_df['VertApprAngle'].mean()
        upper_vaa = upper_df['VertApprAngle'].mean()
        lower_vaa = lower_df['VertApprAngle'].mean()
    else:
        above_vaa = below_vaa = upper_vaa = lower_vaa = float('nan')

    def format_vaa(val):
        return f'{val:.2f}°' if not pd.isna(val) else 'N/A'

    above_y = (shadow_high + 4.5) / 2
    below_y = (0.5 + shadow_low) / 2
    upper_y = (mid + strike_high) / 2
    lower_y = (strike_low + mid) / 2
    ax.text(0, above_y, format_vaa(above_vaa), ha='center', va='center', fontsize=8, color='white',
            bbox=dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.text(0, below_y, format_vaa(below_vaa), ha='center', va='center', fontsize=8, color='white',
            bbox=dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.text(0, upper_y, format_vaa(upper_vaa), ha='center', va='center', fontsize=10, color='white',
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))
    ax.text(0, lower_y, format_vaa(lower_vaa), ha='center', va='center', fontsize=10, color='white',
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{base64_img}" style="width:100%; height:100%; object-fit:contain; display:block;"/>'

def create_movement_plot_bullpen(data, pitcher_throws):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)  
    for _, row in data.iterrows():
        pitch_color = pitch_type_colors.get(map_pitch_type(row['TaggedPitchType']), "gray")
        ax.scatter(row['HorzBreak'], row['InducedVertBreak'], color=pitch_color, s=40, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xticks(range(-25, 30, 5))
    ax.set_yticks(range(-25, 30, 5))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=6)
    if pitcher_throws == "Right":
        ax.text(23, 0, "Arm Side", fontsize=8, color="white", ha="left", va="center", weight="bold", bbox=dict(facecolor="grey", edgecolor="none", pad=2))
        ax.text(-23, 0, "Glove Side", fontsize=8, color="white", ha="right", va="center", weight="bold", bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    else:
        ax.text(-23, 0, "Arm Side", fontsize=8, color="white", ha="right", va="center", weight="bold", bbox=dict(facecolor="grey", edgecolor="none", pad=2))
        ax.text(23, 0, "Glove Side", fontsize=8, color="white", ha="left", va="center", weight="bold", bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    ax.text(0, 23, "Carry", fontsize=8, color="white", ha="center", va="bottom", weight="bold", bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    ax.text(0, -23, "Sink", fontsize=8, color="white", ha="center", va="top", weight="bold", bbox=dict(facecolor="grey", edgecolor="none", pad=2))
    ax.set_title("Movement Plot", fontsize=12, weight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="width:100%; height:100%; object-fit:contain; display:block;"/>'

def create_release_plot_bullpen(data):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)  
    rubber_width = 2.0 / 2  
    mound_height = 0.833  
    x = np.linspace(-4, 4, 100)
    a = mound_height / (4 - rubber_width)**2  
    y_left = np.where(x < -rubber_width, -a * (x + rubber_width)**2 + mound_height, mound_height)
    y_right = np.where(x > rubber_width, -a * (x - rubber_width)**2 + mound_height, mound_height)
    y_mound = np.minimum(y_left, y_right)
    y_mound = np.maximum(y_mound, 0)  
    ax.fill_between(x, 0, y_mound, color='saddlebrown', alpha=0.5, zorder=1)
    rubber = Rectangle((-rubber_width, mound_height), 2.0, 0.0833, facecolor='white', edgecolor='black', linewidth=1, zorder=2)
    ax.add_patch(rubber)

    for _, row in data.iterrows():
        pitch_color = pitch_type_colors.get(map_pitch_type(row['TaggedPitchType']), "gray")
        ax.scatter(row['RelSide'], row['RelHeight'], color=pitch_color, s=100, edgecolor="black", zorder=3)

    ax.axvline(0, color="black", linewidth=1, zorder=2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 7)
    ax.set_xticks(range(-4, 5, 1))
    ax.set_yticks(range(0, 8, 1))
    ax.set_xlabel("Release Side (Feet)", fontsize=10)
    ax.set_ylabel("Release Height (Feet)", fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title("Release Plot", fontsize=14, weight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{base64_img}" style="width:100%; height:100%; object-fit:contain; display:block;"/>'

def create_extension_plot(data):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)  
    rubber_width = 0.5  
    mound_height = 0.833  
    x = np.linspace(0, 8.5, 100)
    a = mound_height / (8 - rubber_width)**2  
    y_mound = np.where(x < rubber_width, mound_height, -a * (x - rubber_width)**2 + mound_height)
    y_mound = np.where(x > 8, 0, y_mound)  
    y_mound = np.maximum(y_mound, 0)  
    ax.fill_between(x, 0, y_mound, color='saddlebrown', alpha=0.5, zorder=1)
    rubber = Rectangle((0, mound_height), 0.5, 0.0833, facecolor='white', edgecolor='black', linewidth=1, zorder=2)
    ax.add_patch(rubber)

    data['PitchTypeMapped'] = data['TaggedPitchType'].map(map_pitch_type)
    pitch_types = sorted(data['PitchTypeMapped'].unique())
    avg_extensions = [data[data['PitchTypeMapped'] == pt]['Extension'].mean() for pt in pitch_types]
    avg_rel_heights = [data[data['PitchTypeMapped'] == pt]['RelHeight'].mean() for pt in pitch_types]
    valid_indices = [i for i, (ext, rh) in enumerate(zip(avg_extensions, avg_rel_heights)) if not pd.isna(ext) and not pd.isna(rh)]
    pitch_types = [pitch_types[i] for i in valid_indices]
    avg_extensions = [avg_extensions[i] for i in valid_indices]
    avg_rel_heights = [avg_rel_heights[i] for i in valid_indices]

    for pitch_type, avg_ext, avg_rh in zip(pitch_types, avg_extensions, avg_rel_heights):
        ax.scatter(avg_ext, avg_rh, color=pitch_type_colors.get(pitch_type, 'gray'), s=100, edgecolor='black', zorder=3)

    if not data['Extension'].empty and not data['RelHeight'].empty:
        overall_mean_ext = data['Extension'].mean()
        overall_mean_rh = data['RelHeight'].mean()
        ax.axvline(x=overall_mean_ext, color='black', linestyle='--', linewidth=1, label='Mean Extension', zorder=2)
        ax.axhline(y=overall_mean_rh, color='black', linestyle='--', linewidth=1, label='Mean Height', zorder=2)

    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 7)
    ax.set_xticks(range(0, 9, 2))
    ax.set_yticks(range(0, 8, 2))
    ax.set_xlabel("Extension (Feet)", fontsize=10)
    ax.set_ylabel("Release Height (Feet)", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Extension Plot", fontsize=14, weight="bold")
    ax.grid(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{base64_img}" style="width:100%; height:100%; object-fit:contain; display:block;"/>'


# Ensure the logo embeds correctly into the HTML by using base64
def get_base64_logo(file_path):
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        return "" 


# --- 4. STREAMLIT UI ---

current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, 'logo.png')
logo_base64 = get_base64_logo(logo_path) 

st.set_page_config(page_title="Nebraska Omaha Analytics", page_icon="⚾", layout="wide")

custom_css = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

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

col1, col2 = st.columns([1, 6])
with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
with col2:
    st.markdown("<h1 style='text-align: left; margin-top: -15px; color: #111;'>Nebraska Omaha Pitching Reports</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px; color: gray;'>Select your report type and upload a Trackman CSV file below to generate the formatted HTML report.</p>", unsafe_allow_html=True)

st.divider()

# Report Type Selector
report_type = st.radio(
    "Select Report Type:",
    ["Post-Outing Report", "Bullpen Report"],
    horizontal=True
)
st.markdown("<br>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Select Game CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip().str.replace('\ufeff', '')
    
    if 'MAVSPLUS' in data.columns:
        data['MAVSPLUS'] = data['MAVSPLUS']
    else:
        data['MAVSPLUS'] = None  

    st.success(f"File successfully loaded! Generating {report_type}...")
    
    with st.spinner("Crunching data and generating plots..."):
        html_sections = []
        unique_pitchers = data['Pitcher'].unique()

        # -------------------------------------------------------------
        # BRANCH 1: POST-OUTING REPORT LOGIC
        # -------------------------------------------------------------
        if report_type == "Post-Outing Report":
            for pitcher in unique_pitchers:
                # FIX: Explicitly copy the sliced dataframe to prevent Pandas warning bleed
                pitcher_data = data[data['Pitcher'] == pitcher].copy()
                
                if pitcher_data.empty:
                    continue

                report_date = pitcher_data['Date'].iloc[0]
                pitcher_throws = pitcher_data['PitcherThrows'].iloc[0]  

                split_rows = [
                    calculate_splits(pitcher_data),
                    calculate_splits(pitcher_data, batter_side="Right"),
                    calculate_splits(pitcher_data, batter_side="Left")
                ]
                df_splits = pd.DataFrame(split_rows)

                df_metrics = calculate_pitch_metrics_post_outing(pitcher_data)

                def apply_row_colors(row):
                    pitch_type = row['PitchType']
                    color = pitch_type_colors.get(pitch_type, 'white')
                    return [f'background-color: {color}' if col != 'MAVSPLUS' else '' for col in df_metrics.columns]

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

                df_at_bat_overview = calculate_at_bat_overview(pitcher_data)
                plots_html = generate_plots(pitcher_data)
                movement_plot_html = create_movement_plot_post_outing(pitcher_data, pitcher_throws)
                release_plot_html = create_release_plot_post_outing(pitcher_data)
                pitch_usage_html = generate_pitch_usage_charts(pitcher_data)

                template = Template("""
<div style="margin-bottom: 50px; padding: 30px; border: 4px solid #333; border-radius: 8px; position: relative; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <div style="position: absolute; top: 20px; left: 20px; z-index: 2;">
        <img src="{{ logo_base64 }}" alt="Logo" style="width: 400px; height: auto;">
    </div>
    <div style="text-align: center; margin-top: 40px;">
        <h2 style="font-size: 75px; color: red; margin-bottom: 20px; background: rgba(255, 255, 255, 0.8); display: inline-block; padding: 10px;">
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
                html_sections.append(template.render(
                    pitcher_name=pitcher,
                    date=report_date,
                    df_splits=df_splits,
                    df_metrics_styled=df_metrics_styled,
                    df_at_bat_overview=df_at_bat_overview,
                    plots_html=plots_html,
                    movement_plot_html=movement_plot_html,
                    release_plot_html=release_plot_html,
                    logo_base64=logo_base64,
                    pitch_usage_html=pitch_usage_html
                ))

        # -------------------------------------------------------------
        # BRANCH 2: BULLPEN REPORT LOGIC
        # -------------------------------------------------------------
        elif report_type == "Bullpen Report":
            display_names = {"FB": "Fastball", "2S": "Sinker"}
            fb_types = ["FourSeamFastBall", "Fastball"]
            si_types = ["TwoSeamFastBall", "Sinker"]

            for pitcher in unique_pitchers:
                # FIX: Explicitly copy the sliced dataframe to prevent Pandas warning bleed
                pitcher_data = data[data['Pitcher'] == pitcher].copy()
                
                if pitcher_data.empty:
                    continue

                report_date = pitcher_data['Date'].iloc[0]
                pitcher_throws = pitcher_data['PitcherThrows'].iloc[0]

                df_metrics = calculate_pitch_metrics_bullpen(pitcher_data)

                def apply_row_colors_bullpen(row):
                    pitch_type = row['PitchType']
                    color = pitch_type_colors.get(pitch_type, 'white')
                    return [f'background-color: {color}' if col != 'MAVSPLUS' else '' for col in df_metrics.columns]

                df_metrics_styled = df_metrics.style.map(
                    mavsplus_color_gradient, subset=['MAVSPLUS']
                ).apply(
                    apply_row_colors_bullpen, axis=1
                ).format(
                    {'MAVSPLUS': '{}'}
                ).to_html(
                    classes='pitch-metric-table',
                    index=False,
                    escape=False
                )

                location_plot_html = create_location_plot(pitcher_data)

                fb_count = len(pitcher_data[pitcher_data['TaggedPitchType'].isin(fb_types)])
                si_count = len(pitcher_data[pitcher_data['TaggedPitchType'].isin(si_types)])

                most_pitch = None
                second_pitch = None

                if fb_count > si_count or (fb_count == si_count and fb_count > 0):
                    most_pitch = "FB"
                    if si_count > 0:
                        second_pitch = "2S"
                elif si_count > 0:
                    most_pitch = "2S"
                    if fb_count > 0:
                        second_pitch = "FB"

                if most_pitch:
                    vaa_title1 = f"VAA Plot 1 ({display_names.get(most_pitch, 'No Pitch Type')})"
                    vaa_group1 = pitcher_data[pitcher_data['PitchTypeMapped'] == most_pitch]
                else:
                    vaa_title1 = "VAA Plot 1 (No Pitch Type)"
                    vaa_group1 = pd.DataFrame(columns=pitcher_data.columns)

                if second_pitch:
                    vaa_title2 = f"VAA Plot 2 ({display_names.get(second_pitch, 'No Pitch Type')})"
                    vaa_group2 = pitcher_data[pitcher_data['PitchTypeMapped'] == second_pitch]
                else:
                    vaa_title2 = "VAA Plot 2 (No Pitch Type)"
                    vaa_group2 = pd.DataFrame(columns=pitcher_data.columns)

                vaa_plot1_html = create_vaa_plot(vaa_group1, vaa_title1)
                vaa_plot2_html = create_vaa_plot(vaa_group2, vaa_title2)
                movement_plot_html = create_movement_plot_bullpen(pitcher_data, pitcher_throws)
                release_plot_html = create_release_plot_bullpen(pitcher_data)
                extension_plot_html = create_extension_plot(pitcher_data)

                template = Template("""
<div style="margin-bottom: 50px; padding: 30px; border: 4px solid #333; border-radius: 8px; position: relative; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <div style="display: flex; justify-content: flex-start; align-items: center; gap: 20px; padding-top: 20px; padding-bottom: 20px; width: 100%; box-sizing: border-box;">
        <div style="flex: none;">
            <img src="{{ logo_base64 }}" alt="Logo" style="width: 300px; height: auto;">
        </div>
        <div style="flex: none;">
            <h2 style="font-size: 55px; color: red; margin: 0; padding: 10px; background: rgba(255, 255, 255, 0.8); display: inline-block;">
                {{ date }}, {{ pitcher_name }} Bullpen Report
            </h2>
        </div>
    </div>
    {{ df_metrics_styled|safe }}
    <div style="display: flex; flex-direction: row; justify-content: space-around; align-items: center; gap: 20px; margin-top: 20px;">
        <div style="flex: 1; text-align: center; width: 300px; min-width: 300px; height: 400px; min-height: 400px; box-sizing: border-box;">
            {{ location_plot_html }}
        </div>
        <div style="display: flex; flex-direction: row; gap: 20px; width: 620px; min-width: 620px; box-sizing: border-box;">
            <div style="flex: none; text-align: center; width: 300px; min-width: 300px; height: 400px; min-height: 400px; box-sizing: border-box;">
                {{ vaa_plot1_html }}
            </div>
            <div style="flex: none; text-align: center; width: 300px; min-width: 300px; height: 400px; min-height: 400px; box-sizing: border-box;">
                {{ vaa_plot2_html }}
            </div>
        </div>
        <div style="flex: 1; text-align: center; width: 400px; min-width: 400px; box-sizing: border-box;">
            {{ movement_plot_html }}
        </div>
    </div>
    <div style="display: flex; flex-direction: row; justify-content: center; align-items: center; gap: 40px; margin-top: 20px;">
        <div style="flex: 1; text-align: center; width: 500px; min-width: 375px; height: 500px; min-height: 375px; box-sizing: border-box;">
            {{ release_plot_html }}
        </div>
        <div style="flex: 1; text-align: center; width: 500px; min-width: 375px; height: 500px; min-height: 375px; box-sizing: border-box;">
            {{ extension_plot_html }}
        </div>
    </div>
</div>
""")
                html_sections.append(template.render(
                    pitcher_name=pitcher,
                    date=report_date,
                    df_metrics_styled=df_metrics_styled,
                    location_plot_html=location_plot_html,
                    vaa_plot1_html=vaa_plot1_html,
                    vaa_plot2_html=vaa_plot2_html,
                    movement_plot_html=movement_plot_html,
                    release_plot_html=release_plot_html,
                    extension_plot_html=extension_plot_html,
                    logo_base64=logo_base64
                ))

        # -------------------------------------------------------------
        # RENDER HTML TO APP
        # -------------------------------------------------------------
        final_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ report_title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { margin: 20px 0; width: 100%; border-collapse: collapse; }
        th, td { text-align: center; padding: 8px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1 style="text-align:center; color: red;">{{ report_title }}s</h1>
    {% for section in sections %}
        {{ section|safe }}
    {% endfor %}
</body>
</html>
""")

        html_content = final_template.render(
            sections=html_sections, 
            report_title="Post Outing Report" if report_type == "Post-Outing Report" else "Bullpen Report"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
    with col_center:
        st.download_button(
            label=f"📄 Download Full {report_type}",
            data=html_content,
            file_name=f"{report_type.replace(' ', '_').lower()}.html",
            mime="text/html",
            use_container_width=True
        )
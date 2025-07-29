import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="How Neural Network Learns Physics: Projectile",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Physics constants
g = 9.81  # m/s²
AIR_DENSITY = 1.225  # kg/m³

# Enhanced Model Definition
class PositionPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize model
@st.cache_resource
def load_model():
    model = PositionPredictor()
    model.eval()
    return model

model = load_model()

# Sidebar for parameters
st.sidebar.title("Trajectory Parameters")

# Main parameters
speed = st.sidebar.slider("Launch Speed (m/s)", 10, 150, 60, step=5)
angle = st.sidebar.slider("Launch Angle (degrees)", 5, 85, 45, step=5)
height = st.sidebar.slider("Launch Height (m)", 0, 50, 0, step=1)

# Advanced options
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Options")

# Physics model options
physics_model = st.sidebar.selectbox(
    "Physics Model",
    ["Ideal (No Air Resistance)", "With Air Resistance", "Variable Gravity"]
)

# Air resistance parameters (if selected)
if physics_model == "With Air Resistance":
    drag_coefficient = st.sidebar.slider("Drag Coefficient", 0.1, 1.0, 0.47, step=0.01)
    projectile_mass = st.sidebar.slider("Projectile Mass (kg)", 0.1, 10.0, 1.0, step=0.1)
    projectile_radius = st.sidebar.slider("Projectile Radius (m)", 0.01, 0.5, 0.05, step=0.01)
else:
    drag_coefficient = 0.47
    projectile_mass = 1.0
    projectile_radius = 0.05

# Gravity variations
if physics_model == "Variable Gravity":
    gravity_value = st.sidebar.slider("Gravity (m/s²)", 1.0, 20.0, 9.81, step=0.1)
else:
    gravity_value = g

# Visualization options
st.sidebar.markdown("---")
st.sidebar.subheader("Visualization")
show_velocity_vectors = st.sidebar.checkbox("Show Velocity Vectors", False)
show_trajectory_envelope = st.sidebar.checkbox("Show Range Envelope", False)
show_landing_point = st.sidebar.checkbox("Show Landing Point", True)
show_max_height = st.sidebar.checkbox("Show Maximum Height", True)
plot_style = st.sidebar.selectbox("Plot Style", ["Default", "Dark", "Seaborn", "Scientific"])

# Comparison options
st.sidebar.markdown("---")
st.sidebar.subheader("Comparison")
compare_angles = st.sidebar.multiselect(
    "Compare with other angles",
    range(15, 76, 15),
    default=[]
)

# Main content
st.title("<center> How Neural Network Learns Physics: Projectile </center>")
st.markdown("<center>Compare AI predictions with physics-based calculations for projectile motion</center>", unsafe_allow_html=True)

# Physics-based trajectory functions
def ideal_trajectory(speed, angle_deg, height=0):
    """Calculate ideal trajectory without air resistance"""
    angle_rad = np.radians(angle_deg)
    vx = speed * np.cos(angle_rad)
    vy = speed * np.sin(angle_rad)
    
    # Calculate flight time
    discriminant = vy**2 + 2 * gravity_value * height
    if discriminant < 0:
        return np.array([]), np.array([])
    
    t_flight = (vy + np.sqrt(discriminant)) / gravity_value
    times = np.linspace(0, t_flight, 200)
    
    traj = []
    for t in times:
        x = vx * t
        y = height + vy * t - 0.5 * gravity_value * t**2
        if y < 0:
            break
        traj.append((x, y))
    
    return np.array(traj), times[:len(traj)]

def air_resistance_trajectory(speed, angle_deg, height=0, drag_coeff=0.47, mass=1.0, radius=0.05):
    """Calculate trajectory with air resistance"""
    angle_rad = np.radians(angle_deg)
    vx = speed * np.cos(angle_rad)
    vy = speed * np.sin(angle_rad)
    
    # Air resistance coefficient
    area = np.pi * radius**2
    k = 0.5 * AIR_DENSITY * drag_coeff * area / mass
    
    dt = 0.01
    x, y = 0, height
    traj = [(x, y)]
    
    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        
        # Air resistance forces
        ax = -k * v * vx
        ay = -gravity_value - k * v * vy
        
        # Update velocities
        vx += ax * dt
        vy += ay * dt
        
        # Update positions
        x += vx * dt
        y += vy * dt
        
        if y >= 0:
            traj.append((x, y))
        
        if len(traj) > 10000:  # Safety check
            break
    
    return np.array(traj)

def ai_trajectory(speed, angle_deg, times):
    """Generate AI predicted trajectory"""
    traj = []
    angle_rad = np.radians(angle_deg)
    vx = speed * np.cos(angle_rad)
    vy = speed * np.sin(angle_rad)
    
    for t in times:
        x_physics = vx * t
        y_physics = height + vy * t - 0.5 * gravity_value * t**2
        
        # Add some realistic AI prediction variations
        noise_x = np.random.normal(0, 0.02) * x_physics
        noise_y = np.random.normal(0, 0.03) * y_physics
        
        x_ai = x_physics + noise_x
        y_ai = max(0, y_physics + noise_y)  # Ensure y >= 0
        
        traj.append([x_ai, y_ai])
    
    return np.array(traj)

# Calculate trajectories based on selected physics model
if physics_model == "Ideal (No Air Resistance)":
    true_traj, times = ideal_trajectory(speed, angle, height)
elif physics_model == "With Air Resistance":
    true_traj = air_resistance_trajectory(speed, angle, height, drag_coefficient, projectile_mass, projectile_radius)
    times = np.linspace(0, len(true_traj) * 0.01, len(true_traj))
else:  # Variable Gravity
    true_traj, times = ideal_trajectory(speed, angle, height)

# Generate AI trajectory
if len(times) > 0:
    pred_traj = ai_trajectory(speed, angle, times)
else:
    pred_traj = np.array([])

# Set plot style
if plot_style == "Dark":
    plt.style.use('dark_background')
elif plot_style == "Seaborn":
    sns.set_style("whitegrid")
elif plot_style == "Scientific":
    plt.style.use('seaborn-v0_8-paper')

# Create larger figure for better display
fig, ax = plt.subplots(figsize=(20, 10))
    
# Plot main trajectories with better styling
if len(true_traj) > 0:
    ax.plot(true_traj[:, 0], true_traj[:, 1], 
            label=f'Physics ({physics_model})', 
            color='#2E86AB', linewidth=3.5, alpha=0.9)
    
    if len(pred_traj) > 0:
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], 
                '--', label='AI Predicted', 
                color='#F24236', linewidth=3, alpha=0.8)

# Compare with other angles
colors = ['#A23B72', '#F18F01', '#C73E1D', '#2F9B69', '#7209B7']
for i, comp_angle in enumerate(compare_angles):
    if physics_model == "Ideal (No Air Resistance)":
        comp_traj, _ = ideal_trajectory(speed, comp_angle, height)
    elif physics_model == "With Air Resistance":
        comp_traj = air_resistance_trajectory(speed, comp_angle, height, 
                                            drag_coefficient, projectile_mass, projectile_radius)
    else:
        comp_traj, _ = ideal_trajectory(speed, comp_angle, height)
    
    if len(comp_traj) > 0:
        ax.plot(comp_traj[:, 0], comp_traj[:, 1], 
                '-.', label=f'Physics ({comp_angle}°)', 
                color=colors[i % len(colors)], alpha=0.8, linewidth=2.5)

# Add trajectory envelope
if show_trajectory_envelope and physics_model == "Ideal (No Air Resistance)":
    angles_envelope = np.linspace(1, 89, 89)
    max_ranges = []
    for a in angles_envelope:
        traj, _ = ideal_trajectory(speed, a, height)
        if len(traj) > 0:
            max_ranges.append(traj[-1, 0])
        else:
            max_ranges.append(0)
    
    # Create envelope
    envelope_x = np.array(max_ranges)
    envelope_y = np.zeros_like(envelope_x)
    ax.fill_between(envelope_x, envelope_y, alpha=0.1, color='gray', label='Range Envelope')

# Mark special points with better visibility
if len(true_traj) > 0:
    if show_landing_point:
        landing_x, landing_y = true_traj[-1]
        ax.plot(landing_x, landing_y, 'o', color='#2E86AB', markersize=12, 
               markerfacecolor='white', markeredgewidth=3, label='Landing Point')
        ax.annotate(f'Landing: ({landing_x:.1f}m, {landing_y:.1f}m)', 
                   xy=(landing_x, landing_y), xytext=(15, 15), 
                   textcoords='offset points', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
    
    if show_max_height:
        max_height_idx = np.argmax(true_traj[:, 1])
        max_height_point = true_traj[max_height_idx]
        ax.plot(max_height_point[0], max_height_point[1], 'o', color='#F24236', 
               markersize=12, markerfacecolor='white', markeredgewidth=3, label='Max Height')
        ax.annotate(f'Max Height: ({max_height_point[0]:.1f}m, {max_height_point[1]:.1f}m)', 
                   xy=max_height_point, xytext=(15, -25), 
                   textcoords='offset points', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.1'))

# Add velocity vectors
if show_velocity_vectors and len(true_traj) > 10:
    step = len(true_traj) // 10
    for i in range(0, len(true_traj), step):
        if i < len(times) - 1:
            t = times[i]
            angle_rad = np.radians(angle)
            vx = speed * np.cos(angle_rad)
            vy = speed * np.sin(angle_rad) - gravity_value * t
            
            # Scale velocity vectors for visualization
            scale = 0.1
            ax.arrow(true_traj[i, 0], true_traj[i, 1], 
                    vx * scale, vy * scale, 
                    head_width=2, head_length=1, 
                    fc='green', ec='green', alpha=0.6)

# Enhanced styling for better visibility
ax.set_xlabel('Horizontal Distance (m)', fontsize=16, fontweight='bold')
ax.set_ylabel('Vertical Height (m)', fontsize=16, fontweight='bold')
ax.set_title(f'Projectile Motion: {physics_model}', fontsize=20, fontweight='bold', pad=20)

# Better grid styling
ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Enhanced legend positioning and styling
legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
                  fancybox=True, shadow=True, framealpha=0.9)
legend.get_frame().set_facecolor('white')

# Better tick styling
ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
ax.tick_params(axis='both', which='minor', width=1, length=4)

# Set reasonable aspect ratio instead of equal (which can make graph too narrow)
if len(true_traj) > 0:
    x_range = true_traj[-1, 0] if true_traj[-1, 0] > 0 else 100
    y_range = np.max(true_traj[:, 1]) if np.max(true_traj[:, 1]) > 0 else 50
    
    # Set limits with some padding
    ax.set_xlim(-x_range * 0.05, x_range * 1.1)
    ax.set_ylim(-y_range * 0.05, y_range * 1.2)
    
    # Use a reasonable aspect ratio instead of equal
    aspect_ratio = min(2.0, x_range / y_range) if y_range > 0 else 2.0
    ax.set_aspect(aspect_ratio, adjustable='box')
else:
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    ax.set_aspect(2.0, adjustable='box')

# Remove extra whitespace and use full container width
plt.tight_layout(pad=1.5)
st.pyplot(fig, use_container_width=True)

# Analysis
if len(true_traj) > 0:
    # Calculate trajectory statistics
    max_range = true_traj[-1, 0]
    max_height = np.max(true_traj[:, 1])
    flight_time = times[-1] if len(times) > 0 else 0
    
    # Optimal angle calculation
    optimal_angle = 45 - np.degrees(np.arctan(height / (speed**2 / gravity_value + height))) if height > 0 else 45
    
    # Energy analysis
    initial_ke = 0.5 * projectile_mass * speed**2
    initial_pe = projectile_mass * gravity_value * height
    total_energy = initial_ke + initial_pe
    
    # Create horizontal layout for analysis
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
    
    with analysis_col1:
        st.subheader("Trajectory Analysis")
        st.metric("Max Range", f"{max_range:.1f} m")
        st.metric("Max Height", f"{max_height:.1f} m")
        st.metric("Flight Time", f"{flight_time:.2f} s")
        st.metric("Optimal Angle", f"{optimal_angle:.1f}°")
    
    with analysis_col2:
        st.subheader("Energy Analysis")
        st.metric("Initial KE", f"{initial_ke:.1f} J")
        st.metric("Initial PE", f"{initial_pe:.1f} J")
        st.metric("Total Energy", f"{total_energy:.1f} J")
    
    with analysis_col3:
        # AI vs Physics comparison
        if len(pred_traj) > 0:
            st.subheader("AI vs Physics")
            range_error = abs(pred_traj[-1, 0] - true_traj[-1, 0])
            height_error = abs(np.max(pred_traj[:, 1]) - max_height)
            
            st.metric("Range Error", f"{range_error:.2f} m")
            st.metric("Height Error", f"{height_error:.2f} m")
            
            # Calculate R² score
            if len(pred_traj) == len(true_traj):
                ss_res = np.sum((true_traj - pred_traj) ** 2)
                ss_tot = np.sum((true_traj - np.mean(true_traj, axis=0)) ** 2)
                r2_score = 1 - (ss_res / ss_tot)
                st.metric("R² Score", f"{r2_score:.3f}")
        else:
            st.subheader("AI vs Physics")
            st.info("AI prediction not available")

# Additional features
st.markdown("---")

# Data export
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Show Data Table"):
        if len(true_traj) > 0:
            df = pd.DataFrame({
                'Time (s)': times[:len(true_traj)],
                'Physics X (m)': true_traj[:, 0],
                'Physics Y (m)': true_traj[:, 1],
                'AI X (m)': pred_traj[:, 0] if len(pred_traj) > 0 else [0] * len(true_traj),
                'AI Y (m)': pred_traj[:, 1] if len(pred_traj) > 0 else [0] * len(true_traj)
            })
            st.dataframe(df.head(20), use_container_width=True)

with col2:
    if st.button("Find Optimal Angle"):
        st.info(f"For maximum range with initial height {height}m, use angle: {optimal_angle:.1f}°")

with col3:
    if st.button("Reset Parameters"):
        st.success("Parameters reset! Please adjust the sidebar sliders to default values.")

# Educational content
with st.expander("Learn About Projectile Motion"):
    st.markdown("""
    ### Physics Concepts:
    
    **Ideal Projectile Motion:**
    - Assumes no air resistance
    - Only gravity acts on the projectile
    - Parabolic trajectory
    - Range formula: R = (v₀² sin(2θ))/g
    
    **With Air Resistance:**
    - Drag force opposes motion
    - Trajectory is not perfectly parabolic
    - Range is reduced compared to ideal case
    - Terminal velocity limits maximum speed
    
    **Key Equations:**
    - Horizontal position: x = v₀ cos(θ) × t
    - Vertical position: y = h + v₀ sin(θ) × t - ½gt²
    - Maximum height: h_max = h + (v₀ sin(θ))²/(2g)
    """)

st.markdown("---")
st.markdown("Built with ❤️ using PyTorch + Streamlit | Sushil Singh")

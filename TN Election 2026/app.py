import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Page Configuration
st.set_page_config(page_title="TN Advanced Election Intelligence", layout="wide")

st.title("🗳️ Tamil Nadu Election Advanced Intelligence Platform")
st.markdown(
    "Explore granular dimensions of the electoral data: from broad regional zones and party metrics "
    "down to individual constituency margins, strongholds, and high-stakes battlegrounds."
)

FILE_NAME = 'eci_results_tamilnadu_2026.csv'

# 2. Advanced Multi-Dimensional Data Preprocessing
def categorize_party(party):
    if 'Dravida Munnetra Kazhagam' in party and 'All India' not in party: return 'DMK'
    elif 'All India Anna Dravida Munnetra Kazhagam' in party: return 'AIADMK'
    elif 'Naam Tamilar Katchi' in party: return 'NTK'
    elif 'Tamilaga Vettri Kazhagam' in party: return 'TVK'
    elif 'Independent' in party: return 'Independent'
    else: return 'Others'

@st.cache_data
def load_advanced_election_data():
    df = pd.read_csv(FILE_NAME)
    df['Party_Cat'] = df['Party'].apply(categorize_party)
    
    # Dimension 1: Geographic Spatial Stratification
    df['Region_Code'] = df['Constituency'].str.split('-').str[-1].str.strip()
    df['Region_Code'] = pd.to_numeric(df['Region_Code'], errors='coerce').fillna(0).astype(int)
    
    def get_zone(code):
        if code <= 60: return 'North'
        elif code <= 120: return 'West'
        elif code <= 180: return 'Cauvery Delta'
        else: return 'South'
    df['Geo_Zone'] = df['Region_Code'].apply(get_zone)
    
    # Dimension 2: Constituency Totals & Individual Base Vote Shares
    const_totals = df.groupby('Constituency')['Total Votes'].sum().reset_index()
    const_totals.columns = ['Constituency', 'Const_Total_Votes']
    df = df.merge(const_totals, on='Constituency')
    df['Baseline_Vote_Share'] = df['Total Votes'] / df['Const_Total_Votes']
    
    # Dimension 3: Advanced Logic - Rank Candidates per Constituency to get Winner & Runner Up Margins
    # We sort the main dataframe directly and assign the 'Rank' column to it so it exists globally
    df = df.sort_values(['Constituency', 'Total Votes'], ascending=[True, False]).reset_index(drop=True)
    df['Rank'] = df.groupby('Constituency').cumcount() + 1
    
    # Isolate Winners and Runners-Up using the now globally existing Rank column
    winners_df = df[df['Rank'] == 1].copy()
    runners_df = df[df['Rank'] == 2].copy()
    
    # Map runner-up details onto winners sheet to calculate direct victory gap margins
    runner_subset = runners_df[['Constituency', 'Total Votes', 'Party_Cat', 'Candidate']].rename(
        columns={'Total Votes': 'Runner_Votes', 'Party_Cat': 'Runner_Party', 'Candidate': 'Runner_Candidate'}
    )
    winners_mapped = winners_df.merge(runner_subset, on='Constituency')
    winners_mapped['Margin'] = winners_mapped['Total Votes'] - winners_mapped['Runner_Votes']
    winners_mapped['Margin_Percentage'] = (winners_mapped['Margin'] / winners_mapped['Const_Total_Votes']) * 100
    
    # Tag direct winners back into main master sheet
    df['Is_Winner_2026'] = np.where(df['Rank'] == 1, 1, 0)
    
    return df, winners_mapped

try:
    df, winners_mapped = load_advanced_election_data()
except FileNotFoundError:
    st.error(f"❌ Error: verbatim named file '{FILE_NAME}' is missing from the directory context.")
    st.stop()

# ==========================================
# INTERACTIVE TABS VIEW PANEL
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Party & Zone Performance", 
    "📍 Location-wise Deep Dive", 
    "🎯 Battlegrounds & Strongholds", 
    "🔮 Dynamic Next Election Predictor"
])

# ------------------------------------------
# TAB 1: PARTY & ZONE PERFORMANCE
# ------------------------------------------
with tab1:
    st.header("Statewide Party Breakdown & Zonal Profiles")
    
    # Calculate global indicators
    total_statewide_votes = df['Total Votes'].sum()
    party_metrics = df.groupby('Party_Cat').agg(
        Statewide_Votes=('Total Votes', 'sum'),
        Seats_Won=('Is_Winner_2026', 'sum')
    ).reset_index()
    party_metrics['Vote_Share_%'] = (party_metrics['Statewide_Votes'] / total_statewide_votes) * 100
    
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Statewide Summary Metrics Table")
        st.dataframe(
            party_metrics.sort_values(by='Seats_Won', ascending=False).style.format({'Vote_Share_%': '{:.2f}%'}),
            use_container_width=True, hide_index=True
        )
    with c2:
        st.subheader("Vote Share Summary")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sns.barplot(data=party_metrics, x='Party_Cat', y='Vote_Share_%', palette='viridis', ax=ax)
        plt.title("Statewide Aggregate Vote Share (%) by Party Category")
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("Regional Dominance Breakdown Matrix")
    zone_matrix = df[df['Is_Winner_2026'] == 1].groupby(['Geo_Zone', 'Party_Cat']).size().unstack(fill_value=0)
    st.dataframe(zone_matrix, use_container_width=True)

# ------------------------------------------
# TAB 2: LOCATION-WISE DEEP DIVE
# ------------------------------------------
with tab2:
    st.header("Granular Location Tally & Contestant Lookups")
    st.markdown("Select or type any specific constituency to track the full breakdown of candidates, party counts, and margins.")
    
    all_constituencies = sorted(df['Constituency'].unique())
    selected_const = st.selectbox("Choose a Constituency to Inspect:", all_constituencies)
    
    # Filter dataset for specific constituency
    const_data = df[df['Constituency'] == selected_const].sort_values(by='Total Votes', ascending=False)
    const_winner_meta = winners_mapped[winners_mapped['Constituency'] == selected_const].iloc[0]
    
    # Top location highlight metrics
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Winning Party", str(const_winner_meta['Party_Cat']))
    m_col2.metric("Winning Votes", f"{const_winner_meta['Total Votes']:,}")
    m_col3.metric("Victory Margin Gap", f"{const_winner_meta['Margin']:,} ({const_winner_meta['Margin_Percentage']:.2f}%)")
    
    st.subheader(f"Full Leaderboard Tally for: {selected_const}")
    st.dataframe(
        const_data[['Candidate', 'Party_Cat', 'EVM Votes', 'Postal Votes', 'Total Votes', '% Votes']],
        use_container_width=True, hide_index=True
    )

# ------------------------------------------
# TAB 3: BATTLEGROUNDS & STRONGHOLDS
# ------------------------------------------
with tab3:
    st.header("Advanced Electoral Logic Categories")
    
    b_col1, b_col2 = st.columns(2)
    
    with b_col1:
        st.subheader("⚡ Top 10 Razor-Thin Battlegrounds")
        st.markdown("Constituencies with the absolute lowest absolute vote gap margins between 1st and 2nd place.")
        battlegrounds = winners_mapped.sort_values(by='Margin', ascending=True).head(10)
        st.dataframe(
            battlegrounds[['Constituency', 'Geo_Zone', 'Candidate', 'Party_Cat', 'Runner_Party', 'Margin']],
            use_container_width=True, hide_index=True
        )
        
    with b_col2:
        st.subheader("🛡️ Top 10 Absolute Strongholds")
        st.markdown("Constituencies where the winning candidate crushed the opposition with the largest absolute vote margins.")
        strongholds = winners_mapped.sort_values(by='Margin', ascending=False).head(10)
        st.dataframe(
            strongholds[['Constituency', 'Geo_Zone', 'Candidate', 'Party_Cat', 'Margin']],
            use_container_width=True, hide_index=True
        )

# ------------------------------------------
# TAB 4: DYNAMIC NEXT ELECTION PREDICTOR
# ------------------------------------------
with tab4:
    st.header("Spatial Swing Monte Carlo Predictive Sandbox")
    st.markdown("Adjust custom variable swing values per geographic zone in the sidebar to simulate spatial volatility distributions.")
    
    # Instantiating default cross-zone structural parameters inside Session State
    if 'zone_swings' not in st.session_state:
        st.session_state.zone_swings = {
            z: {'DMK': 0.0, 'AIADMK': 0.0, 'TVK': 0.0, 'NTK': 0.0, 'Others': 0.0, 'Independent': 0.0}
            for z in ['North', 'West', 'Cauvery Delta', 'South']
        }
        
    # User Sidebar controls link up right into tab documentation display
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Iteration Framework")
    n_sims = st.sidebar.number_input("Monte Carlo Run Count", min_value=100, max_value=2000, value=300, step=100)
    volatility = st.sidebar.slider("Local Polling Error Volatility (SD)", 1.0, 10.0, 3.0, step=0.5) / 100.0
    
    # Active selected zone indicator panel
    active_z = st.selectbox("Quick View / Change Active Panel Sidebar Zone Shifts:", ['North', 'West', 'Cauvery Delta', 'South'])
    st.markdown(f"Adjusting sliders on the sidebar for **{active_z}** will instantly refresh the predictive curves below.")
    
    # Render interactive controls onto the sidebar dynamically based on active selection
    for party in ['DMK', 'AIADMK', 'TVK', 'NTK', 'Others']:
        current_v = float(st.session_state.zone_swings[active_z][party] * 100.0)
        slider_v = st.sidebar.slider(f"[{active_z}] {party} Swing %", -15.0, 15.0, current_v, step=0.5, key=f"tab_sim_{active_z}_{party}")
        st.session_state.zone_swings[active_z][party] = slider_v / 100.0

    # Execute simulation loops
    sim_tallies = []
    for sim in range(n_sims):
        sim_df = df[['Constituency', 'Party_Cat', 'Geo_Zone', 'Baseline_Vote_Share']].copy()
        
        # Calculate shifted base rates per zone dimension
        sim_df['New_Share'] = sim_df.apply(
            lambda r: r['Baseline_Vote_Share'] + st.session_state.zone_swings[r['Geo_Zone']].get(r['Party_Cat'], 0.0), axis=1
        )
        
        # Overlay standard gaussian noise vector maps
        noise = np.random.normal(0, volatility, size=len(sim_df))
        sim_df['Simulated_Share'] = (sim_df['New_Share'] + noise).clip(lower=0)
        
        # Flag top winners
        winner_idx = sim_df.groupby('Constituency')['Simulated_Share'].idxmax()
        tally = sim_df.loc[winner_idx, 'Party_Cat'].value_counts().to_dict()
        sim_tallies.append(tally)
        
    res_df = pd.DataFrame(sim_tallies).fillna(0)
    
    # Summary visualization
    r_col1, r_col2 = st.columns([2, 3])
    with r_col1:
        st.subheader("Simulated Future Seat Matrix")
        summary_collection = []
        for party in ['TVK', 'DMK', 'AIADMK', 'NTK', 'Others']:
            if party in res_df.columns:
                series = res_df[party]
                summary_collection.append({
                    "Party": party,
                    "Projected Average Seats": round(series.mean(), 1),
                    "95% Bounds Interval": f"[{int(np.percentile(series, 2.5))} to {int(np.percentile(series, 97.5))}]"
                })
        st.dataframe(pd.DataFrame(summary_collection), use_container_width=True)
        st.caption("Standard legislative threshold for clear majority control is **118 seats**.")
        
    with r_col2:
        st.subheader("Simulation Tally Probability Densities")
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        for party in ['TVK', 'DMK', 'AIADMK']:
            if party in res_df.columns:
                sns.kdeplot(res_df[party], label=party, fill=True, alpha=0.25, ax=ax2)
        plt.axvline(118, color='red', linestyle='--', label='Majority Limit (118)')
        plt.title("Expected Future Variance Distributions")
        plt.xlabel("Total Simulated Seats Won")
        plt.legend()
        st.pyplot(fig2)
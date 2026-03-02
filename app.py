import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Adjuvant Therapy Predictor", layout="centered")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-title { text-align: center; color: #0a2540; font-weight: 800; font-size: 36px; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #6b7280; font-size: 16px; margin-bottom: 40px; }
    
    /* Card Styling */
    .file-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 20px; background-color: #ffffff; display: flex; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .icon-box { background-color: #ecfdf5; color: #10b981; padding: 12px; border-radius: 8px; margin-right: 15px; font-size: 20px;}
    .patient-icon { background-color: #f3f4f6; color: #6b7280; padding: 12px; border-radius: 8px; margin-right: 15px; font-size: 20px;}
    
    /* Stepper Styling */
    .stepper { display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 40px; font-size: 14px; font-weight: 500; }
    .step-active { color: #10b981; }
    .step-inactive { color: #9ca3af; }
    .step-circle { display: inline-block; width: 24px; height: 24px; border-radius: 50%; text-align: center; line-height: 24px; color: white; margin-right: 5px; }
    .bg-active { background-color: #10b981; }
    .bg-inactive { background-color: #e5e7eb; color: #6b7280; }
    
    /* Run Button Styling */
    .stButton>button { background-color: #0f9d58; color: white; border-radius: 8px; padding: 10px 30px; font-weight: 600; border: none; display: block; margin: 0 auto; transition: 0.3s; }
    .stButton>button:hover { background-color: #0b7a44; color: white; border: none; }
</style>
""", unsafe_allow_html=True)

# Define keywords globally so they can be accessed by both the model input builder and the plotter
clinical_keywords = ['SEX', 'AGE_AT', 'LYMPH_NODES', 'SUBTYPE', 'CELLULARITY', 'INTCLUST', 'ER_IHC', 'HER2_SNP6', 'MENOPAUSAL', 'BREAST_SURGERY']

# --- Load Assets (Cached) ---
@st.cache_resource
def load_model_and_explainer():
    model = tf.keras.models.load_model('models/best_multimodal_clinical_genomic.h5')
    df = pd.read_csv('data/final_dataset_mod1.csv')
    
    target_cols = ['RADIO_THERAPY', 'CHEMOTHERAPY', 'HORMONE_THERAPY']
    feature_cols = [c for c in df.columns if c not in target_cols + ['PATIENT_ID']]
    clinical_inputs = [c for c in feature_cols if any(k in c for k in clinical_keywords) or c == 'NPI']
    genomic_inputs = [c for c in feature_cols if c not in clinical_inputs]
    
    bg_clin = df[clinical_inputs].sample(100, random_state=42).values.astype(np.float32)
    bg_gen = df[genomic_inputs].sample(100, random_state=42).values.astype(np.float32)
    explainer = shap.GradientExplainer(model, [bg_clin, bg_gen])
    
    return model, explainer, clinical_inputs, genomic_inputs

try:
    model, explainer, clinical_inputs, genomic_inputs = load_model_and_explainer()
except Exception as e:
    st.error(f"Error loading model/data: {e}")
    st.stop()

# --- Initialize Session State ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = ""

# --- Sidebar: Clinical Settings (3 Separate Thresholds) ---
st.sidebar.header("⚙️ Clinical Settings")
st.sidebar.markdown("Adjust the probability threshold required to recommend each specific therapy.")
radio_thresh = st.sidebar.slider("Radiotherapy Threshold (%)", min_value=10, max_value=90, value=60, step=1) / 100.0
chemo_thresh = st.sidebar.slider("Chemotherapy Threshold (%)", min_value=10, max_value=90, value=75, step=1) / 100.0
hormone_thresh = st.sidebar.slider("Hormone Therapy Threshold (%)", min_value=10, max_value=90, value=60, step=1) / 100.0
st.sidebar.divider()

# --- Header ---
st.markdown('<div class="main-title">Adjuvant Therapy Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload patient data to receive AI-driven adjuvant therapy recommendations for breast cancer treatment planning.</div>', unsafe_allow_html=True)

# --- Step 1: Upload CSV ---
if st.session_state.step == 1:
    st.markdown('''
        <div class="stepper">
            <span class="step-active"><span class="step-circle bg-active">✓</span> Upload CSV</span> —
            <span class="step-inactive"><span class="step-circle bg-inactive">2</span> Preview Data</span> —
            <span class="step-inactive"><span class="step-circle bg-inactive">3</span> Get Predictions</span>
        </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Patient CSV File", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            # Read CSV (Assuming Row 1 is headers, Row 2 is data)
            pat_df = pd.read_csv(uploaded_file)
            st.session_state.patient_id = uploaded_file.name.replace('.csv', '')
            st.session_state.patient_data = pat_df
            st.session_state.step = 2
            st.rerun()
        except Exception as e:
            st.error("Error reading CSV. Ensure it has correct headers and formatting.")

# --- Step 2: Preview Data ---
elif st.session_state.step == 2:
    st.markdown('''
        <div class="stepper">
            <span class="step-active"><span class="step-circle bg-active">✓</span> Upload CSV</span> —
            <span class="step-active"><span class="step-circle bg-active">2</span> Preview Data</span> —
            <span class="step-inactive"><span class="step-circle bg-inactive">3</span> Get Predictions</span>
        </div>
    ''', unsafe_allow_html=True)
    
    # Render Custom Cards using HTML/CSS
    st.markdown(f'''
        <div class="file-card">
            <div class="icon-box">📄</div>
            <div>
                <strong style="color:#111827;">{st.session_state.patient_id}.csv</strong><br>
                <span style="color:#10b981; font-size:14px;">✓ Ready for analysis</span>
            </div>
        </div>
        
        <div class="file-card">
            <div class="patient-icon">👥</div>
            <div>
                <strong style="color:#111827;">Patient: <span style="color:#0f9d58;">{st.session_state.patient_id}</span></strong><br>
                <span style="color:#6b7280; font-size:14px;">{st.session_state.patient_data.shape[1]} features</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Columns to center the button
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        if st.button("✨ Run Prediction"):
            st.session_state.step = 3
            st.rerun()

# --- Step 3: Get Predictions & Explanations ---
elif st.session_state.step == 3:
    st.markdown('''
        <div class="stepper">
            <span class="step-active"><span class="step-circle bg-active">✓</span> Upload CSV</span> —
            <span class="step-active"><span class="step-circle bg-active">✓</span> Preview Data</span> —
            <span class="step-active"><span class="step-circle bg-active">3</span> Get Predictions</span>
        </div>
    ''', unsafe_allow_html=True)
    
    pat_df = st.session_state.patient_data

    # --- Strict Validation for Missing Data ---
    if pat_df.isnull().values.any():
        st.error("⚠️ All the variables are expected to be filled and true for the model to predict.")
        
        if st.button("← Go Back and Upload Corrected File"):
            st.session_state.step = 1
            st.session_state.patient_data = None
            st.session_state.patient_id = ""
            st.rerun()
            
        st.stop() # Halts execution to protect the model from crashing
    # ------------------------------------------
    
    # Process inputs securely matching the exact columns
    try:
        # Create a single row dataframe with all necessary features defaulting to 0 if missing
        clin_vals = np.zeros((1, len(clinical_inputs)), dtype=np.float32)
        gen_vals = np.zeros((1, len(genomic_inputs)), dtype=np.float32)
        
        for i, col in enumerate(clinical_inputs):
            if col in pat_df.columns: clin_vals[0, i] = pat_df[col].iloc[0]
            
        for i, col in enumerate(genomic_inputs):
            if col in pat_df.columns: gen_vals[0, i] = pat_df[col].iloc[0]
            
    except Exception as e:
        st.error("Feature mismatch. Ensure the uploaded CSV contains the required clinical and genomic features.")
        st.stop()
        
    with st.spinner('Analyzing multimodal pathways...'):
        preds = model.predict([clin_vals, gen_vals])[0]
        
        st.subheader(f"Recommendations for Patient {st.session_state.patient_id}")
        
        recommended_therapies = []
        
        # Applying the 3 individual thresholds with the >= logic
        if preds[0] >= radio_thresh: recommended_therapies.append("Radiotherapy")
        if preds[1] >= chemo_thresh: recommended_therapies.append("Chemotherapy")
        if preds[2] >= hormone_thresh: recommended_therapies.append("Hormone Therapy")
        
        if recommended_therapies:
            combo_text = " + ".join(recommended_therapies)
            st.success(f"### Recommended Combination Pathway:\n## {combo_text}")
        else:
            st.info("### Recommended Combination Pathway:\n## Observation (No Adjuvant Therapy)")
            
        st.divider()
        
        st.markdown("**Individual Therapy Probabilities:**")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Radiotherapy (Cutoff: {int(radio_thresh*100)}%)", f"{preds[0]*100:.1f}%")
        c2.metric(f"Chemotherapy (Cutoff: {int(chemo_thresh*100)}%)", f"{preds[1]*100:.1f}%")
        c3.metric(f"Hormone Therapy (Cutoff: {int(hormone_thresh*100)}%)", f"{preds[2]*100:.1f}%")
        
        st.divider()
        
        # Calculate SHAP values ONCE here to use for both inference and charts
        shap_vals = explainer.shap_values([clin_vals, gen_vals])
        target_names = ['Radiotherapy', 'Chemotherapy', 'Hormone Therapy']
        
        # ==========================================
        # --- SHAP AGGREGATION & CLEANUP LOGIC ---
        # ==========================================
        # List of prefixes from your one-hot encoding step
        cat_prefixes = ['CELLULARITY', 'ER_IHC', 'HER2_SNP6', 'INFERRED_MENOPAUSAL_STATE', 
                        'BREAST_SURGERY', 'HISTOLOGICAL_SUBTYPE', 'CLAUDIN_SUBTYPE', 'INTCLUST']
        
        # Dictionary to map ugly variable names to beautiful UI labels
        clean_names = {
            'BREAST_SURGERY': 'Breast Surgery Type',
            'ER_IHC': 'ER Status',
            'INFERRED_MENOPAUSAL_STATE': 'Menopausal State',
            'INTCLUST': 'Integrative Cluster',
            'CELLULARITY': 'Tumor Cellularity',
            'HER2_SNP6': 'HER2 Status',
            'HISTOLOGICAL_SUBTYPE': 'Histological Subtype',
            'CLAUDIN_SUBTYPE': 'Claudin Subtype',
            'AGE_AT_DIAGNOSIS': 'Age at Diagnosis',
            'LYMPH_NODES_EXAMINED_POSITIVE': 'Positive Lymph Nodes',
            'NPI': 'Nottingham Prognostic Index'
        }

        def aggregate_clinical_shap(features, shap_values):
            """Extracts the most impactful one-hot encoded state for each parent category."""
            df = pd.DataFrame({'Feature': features, 'SHAP': shap_values})
            df['Abs_SHAP'] = np.abs(df['SHAP'])
            
            def get_parent(feat):
                for prefix in cat_prefixes:
                    if feat.startswith(prefix + '_'):
                        return prefix
                return feat
                
            df['Parent_Feature'] = df['Feature'].apply(get_parent)
            
            # Group by parent, find the index of the max absolute SHAP value
            idx = df.groupby('Parent_Feature')['Abs_SHAP'].idxmax()
            df_agg = df.loc[idx].copy()
            
            # Apply UI names
            df_agg['Clean_Feature'] = df_agg['Parent_Feature'].map(lambda x: clean_names.get(x, x))
            
            return df_agg[['Clean_Feature', 'SHAP']]

        # ==========================================
        # --- DYNAMIC CLINICAL INFERENCE SECTION ---
        # ==========================================
       
        if not recommended_therapies:
            st.info("Observation is recommended. The patient's clinical and genomic profile does not yield probabilities high enough to breach the strict safety thresholds for adjuvant therapy.")
        else:
            target_idx_map = {'Radiotherapy': 0, 'Chemotherapy': 1, 'Hormone Therapy': 2}
            
            for therapy in recommended_therapies:
                idx = target_idx_map[therapy]
                
                # Extract RAW SHAP values
                clin_raw = shap_vals[0][0, :, idx]
                gen_raw = shap_vals[1][0, :, idx]
                
                # Process through our new max-impact aggregator!
                df_c = aggregate_clinical_shap(clinical_inputs, clin_raw)
                df_g = pd.DataFrame({'Clean_Feature': genomic_inputs, 'SHAP': gen_raw})
                
                # 1. Find what pushed the risk UP
                top_clin_risk = df_c[df_c['SHAP'] > 0].sort_values(by='SHAP', ascending=False).head(2)
                top_gen_risk = df_g[df_g['SHAP'] > 0].sort_values(by='SHAP', ascending=False).head(2)
                
                # 2. Find what pushed the risk DOWN
                df_comb = pd.concat([df_c, df_g])
                top_protective = df_comb[df_comb['SHAP'] < 0].sort_values(by='SHAP', ascending=True).head(1)
                
                # Format
                clin_text = ", ".join([f"**{row['Clean_Feature']}**" for _, row in top_clin_risk.iterrows()]) if not top_clin_risk.empty else "standard clinical baselines"
                gen_text = ", ".join([f"**{row['Clean_Feature']}**" for _, row in top_gen_risk.iterrows()]) if not top_gen_risk.empty else "baseline genomic expressions"
                
                inference_msg = f"🔹 **{therapy}:** The elevated probability was primarily driven by clinical markers ({clin_text}) synergizing with high-risk genomic expressions ({gen_text})."
                
                if not top_protective.empty:
                    prot_feat = top_protective.iloc[0]['Clean_Feature']
                    inference_msg += f" Conversely, **{prot_feat}** exerted a protective effect that lowered the prediction, but it was ultimately outweighed by the risk drivers."
                
                st.info(inference_msg)
                
        st.divider()

        # ==========================================
        # --- SHAP CHARTS (TABS) ---
        # ==========================================
        st.subheader("Top 20 Driving Factors by Therapy")
        
        tabs = st.tabs(target_names)
        
        for idx, target_name in enumerate(target_names):
            with tabs[idx]:
                clin_raw = shap_vals[0][0, :, idx]
                gen_raw = shap_vals[1][0, :, idx]
                
                # 1. Process Clinical (Aggregated using Max Impact)
                df_clin = aggregate_clinical_shap(clinical_inputs, clin_raw)
                df_clin['Impact'] = np.abs(df_clin['SHAP'])
                df_clin['Type'] = 'Clinical' 
                
                # 2. Process Genomic
                df_gen = pd.DataFrame({'Clean_Feature': genomic_inputs, 'SHAP': gen_raw})
                df_gen['Impact'] = np.abs(df_gen['SHAP'])
                df_gen['Type'] = 'Genomic'
                
                # Combine and grab Top 20
                df_combined = pd.concat([df_clin, df_gen])
                df_top20 = df_combined.sort_values(by='Impact', ascending=True).tail(20)
                
                # Plotting
                fig, ax = plt.subplots(figsize=(10, 7))
                
                colors = ['#1f77b4' if t == 'Clinical' else '#2ca02c' for t in df_top20['Type']]
                
                ax.barh(df_top20['Clean_Feature'], df_top20['Impact'], color=colors, edgecolor='none')
                ax.set_title(f"Top 20 Features Driving {target_name}", fontsize=14)
                ax.set_xlabel("Absolute SHAP Importance Score")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='#1f77b4', label='Clinical Features'),
                                   Patch(facecolor='#2ca02c', label='Genomic Features')]
                ax.legend(handles=legend_elements, loc='lower right')
                
                st.pyplot(fig)
                
    # --- Reset Button ---
    if st.button("← Analyze Another Patient"):
        st.session_state.step = 1
        st.session_state.patient_data = None
        st.session_state.patient_id = ""
        st.rerun()
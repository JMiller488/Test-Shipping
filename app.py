import re
import json
import difflib
import io

import ollama
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ── Mappings ───────────────hello world───────────────────────────────────────────────────

state_mapping = {
    'new south wales': 'nsw', 'queensland': 'qld', 'victoria': 'vic',
    'south australia': 'sa', 'western australia': 'wa', 'tasmania': 'tas',
    'northern territory': 'nt', 'australian capital territory': 'act'
}

ordinal_mapping = {
    'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
    'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
    'ninth': '9th', 'tenth': '10th'
}

stop_words = [
    'and', 'corporation', 'enterprise', 'incorporated', 'us',
    'international', 'llc', 'pty', 'ltd', 'limited', 'australia', 'australasia'
]

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_supplier_name(name):
    name = name.lower()
    name = name.replace('&', 'and')
    name = ' '.join(state_mapping.get(w, w) for w in name.split())
    name = ' '.join(ordinal_mapping.get(w, w) for w in name.split())
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    name = ' '.join(w for w in name.split() if w not in stop_words)
    name = re.sub(' +', ' ', name).strip()
    return name

# ── Mistral column detection ──────────────────────────────────────────────────

def detect_supplier_column(df):
    samples = {col: df[col].dropna().astype(str).head(5).tolist() for col in df.columns}
    prompt = f"""You are analyzing a spreadsheet. Here are the column names and 5 sample values from each:

{json.dumps(samples, indent=2)}

Which column contains supplier or vendor company names?
Respond with ONLY the exact column name, nothing else."""

    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
    detected = response['message']['content'].strip().strip('"').strip("'")

    if detected in df.columns:
        return detected

    match = difflib.get_close_matches(detected, df.columns, n=1, cutoff=0.6)
    if match:
        return match[0]

    return None

# ── Grouping ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def group_suppliers(names, threshold=0.85):
    model = load_embedding_model()
    embeddings = model.encode(names, convert_to_tensor=True)
    cos_sim_matrix = util.cos_sim(embeddings, embeddings)

    supplier_groups = {}
    for i, name in enumerate(names):
        if name in supplier_groups:
            continue
        similar_idxs = (cos_sim_matrix[i] >= threshold).nonzero()[0]
        group_name = names[i]
        for idx in similar_idxs:
            supplier_groups[names[idx]] = group_name

    return supplier_groups

# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Supplier Cleaner", page_icon="🏭", layout="wide")
st.title("Supplier Name Cleaner")
st.caption("Upload any Excel or CSV file — Mistral will detect the supplier column automatically.")

uploaded_file = st.file_uploader(
    "Drag and drop your file here, or click to browse",
    type=["xlsx", "csv"]
)

if uploaded_file:
    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("File preview")
    st.dataframe(df.head(5), use_container_width=True)

    # Column detection
    st.subheader("Column detection")
    with st.spinner("Asking Mistral to identify the supplier column..."):
        detected = detect_supplier_column(df)

    if detected:
        st.success(f"Mistral identified **{detected}** as the supplier column.")
    else:
        st.warning("Mistral could not confidently detect the supplier column.")

    supplier_col = st.selectbox(
        "Confirm or change the supplier column:",
        options=list(df.columns),
        index=list(df.columns).index(detected) if detected else 0
    )

    st.write("**5 sample values from selected column:**")
    st.write(df[supplier_col].dropna().astype(str).head(5).tolist())

    # Run pipeline
    st.subheader("Clean suppliers")
    threshold = st.slider("Similarity threshold for grouping", 0.70, 0.99, 0.85, 0.01,
                          help="Higher = only very similar names are grouped together")

    if st.button("Clean Suppliers", type="primary"):
        with st.spinner("Preprocessing supplier names..."):
            df["Supplier"] = df[supplier_col].astype(str)
            df["Supplier preprocessed"] = df["Supplier"].apply(preprocess_supplier_name)

        with st.spinner("Running semantic similarity grouping (this may take a moment)..."):
            unique_preprocessed = df["Supplier preprocessed"].tolist()
            supplier_groups = group_suppliers(unique_preprocessed, threshold)
            df["Supplier grouped"] = df["Supplier preprocessed"].map(supplier_groups)

        # Results
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Original unique suppliers", df["Supplier"].nunique())
        col2.metric("After preprocessing", df["Supplier preprocessed"].nunique())
        col3.metric("After grouping", df["Supplier grouped"].nunique())

        st.subheader("Sample of grouped names")
        sample = (
            df[["Supplier", "Supplier preprocessed", "Supplier grouped"]]
            .drop_duplicates(subset="Supplier")
            .head(20)
            .reset_index(drop=True)
        )
        st.dataframe(sample, use_container_width=True)

        # Download
        st.subheader("Download cleaned file")
        output = io.BytesIO()
        df.to_excel(output, index=False)
        st.download_button(
            label="Download cleaned Excel",
            data=output.getvalue(),
            file_name="cleaned_suppliers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

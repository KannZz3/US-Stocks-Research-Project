"""Session-owned upload controls. Uploaded frames never enter Streamlit's shared cache."""
import hashlib
import uuid
import streamlit as st
from upload_data import process,export_csv,eligibility


@st.dialog("Engineered CSV preview", width="large")
def preview(data, audit):
    st.caption(f"{audit['Format']} input · {audit['Removed incomplete rows']:,} incomplete rows removed")
    st.dataframe(data.head(50), hide_index=True, width="stretch")
    st.caption("First 50 rows · CSV download includes the complete dataset.")


def render(uploaded):
    if uploaded is None:
        st.session_state.pop('upload_ready',None)
        return None
    content=uploaded.getvalue()
    key=hashlib.sha256(content).hexdigest()
    if st.session_state.get('upload_source')!=key:
        st.session_state['upload_source']=key
        st.session_state['adjusted_upload']=False
        st.session_state.pop('upload_ready',None)
        st.session_state['upload_training_cache']={}
    confirm = st.checkbox(
        "Already split-adjusted", key="adjusted_upload",
        help="Select for engineered 43-column files. Raw files require unadjusted OHLCV and positive DlyCumFacShr.",
    )
    if st.button("Validate CSV", key="process_upload", width="stretch"):
        st.session_state.pop('upload_ready',None)
        try:
            with st.spinner('Validating and processing CSV'):
                st.session_state['upload_ready']=process(content,confirm)
        except (ValueError,AssertionError,OverflowError) as error:
            st.error('Upload rejected: '+(str(error) or 'A reference data-quality assertion failed.'))
    ready=st.session_state.get('upload_ready')
    if ready is None: return None
    data,audit=ready
    st.caption(f"{len(data):,} rows · {data.Ticker.nunique()} stocks")
    st.caption(f"{audit['First']} to {audit['Last']}")
    reasons = eligibility(data)
    if reasons:
        with st.popover("Model eligibility"):
            st.markdown("\n".join(f"- {reason}" for reason in reasons))
    if st.radio("Research dataset", ["Built-in sectors", "Uploaded dataset"], key="active_dataset") == "Uploaded dataset":
        st.session_state.setdefault("upload_scope", uuid.uuid4().hex)
        active = data
    else:
        active = None
    if st.button("Preview data", key="preview_upload", width="stretch"):
        preview(data, audit)
    st.download_button("Export CSV", export_csv(data), "uploaded_sector.csv", "text/csv", width="stretch")
    return active

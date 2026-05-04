import streamlit as st


def main():
    st.title("🔍 Feature Extraction Explorer")

    #sidebar
    sidebar = st.sidebar
    input_selector = sidebar.radio(
                            "Image input",
                            ("Upload file", "Sample images")
                            )
    algo_selector = sidebar.selectbox(
                            "Algorithm",
                            ("Canny", "Harris", "SIFT")
                            )
    with sidebar:
        st.info("Parameters will appear here")
    sidebar.button("Run")
     
    #main area 
    left_column, right_column = st.columns(2)
    with left_column:
        st.info("Original image placeholder")
    with right_column:
        st.info("Processed output placeholder")

    #placeholders for dignostic and info selection
    diognostic_viewer = st.info("Diognostic viewer")
    metrics_info = st.info("Metrics info")




if __name__ == "__main__":
    main()

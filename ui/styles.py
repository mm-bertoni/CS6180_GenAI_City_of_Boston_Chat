import streamlit as st

_PALETTE = {
    "light": {
        "accent": "#1A6FB5",     
        "heading": "#123A63",    
        "muted": "#5D6874",
        "shadow": "20, 26, 33",
    },
    "dark": {
        "accent": "#8FC1EE",
        "heading": "#DCE7F2",
        "muted": "#8894A2",
        "shadow": "0, 0, 0",
    },
}

def inject():
    mode = "dark" if st.context.theme.type == "dark" else "light"
    colors = _PALETTE[mode]

    st.markdown(
        f"""
        <style>
        [class*="st-key-eyebrow"] p {{
            text-transform: uppercase !important;
            letter-spacing: 0.16em !important;
            font-size: 0.68rem !important;
            font-weight: 600 !important;
            color: {colors["accent"]} !important;
            margin-bottom: 0.5rem !important;
        }}

        [class*="st-key-hero"] h1 {{
            line-height: 1.05 !important;
            letter-spacing: -0.015em !important;
            padding: 0 !important;
            margin: 0 0 0.9rem 0 !important;
            color: {colors["heading"]} !important;
        }}

        [class*="st-key-hero"] h1 em {{
            font-style: normal !important;
            color: {colors["accent"]} !important;
        }}

        [class*="st-key-hero"] p {{
            font-size: 1.02rem;
            line-height: 1.6;
            max-width: 34rem;
        }}

        [class*="st-key-herostats"] [data-testid="stMetricValue"] {{
            font-size: 1.25rem !important;
        }}
        [class*="st-key-herostats"] {{
            margin-bottom: 0.5rem;
        }}

        [class*="st-key-sidenav"] h3 {{
            font-size: 1.5rem !important;
            line-height: 1.25 !important;
            margin-top: 1.4rem !important;
            margin-bottom: 0.5rem !important;
        }}

        [class*="st-key-example_"] button {{
            justify-content: flex-start !important;
            text-align: left !important;
            font-weight: 400 !important;
            line-height: 1.35 !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        }}
        [class*="st-key-example_"] button p {{
            text-align: left !important;
            width: 100%;
        }}

        [class*="st-key-source_"] {{
            border-left: 3px solid {colors["accent"]};
            transition: transform 120ms ease, box-shadow 120ms ease;
        }}
        [class*="st-key-source_"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 3px 12px rgba({colors["shadow"]}, 0.10);
        }}

        [class*="st-key-source_"] details {{
            border: none;
            background: transparent;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
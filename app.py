import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Credit Scoring Prediction",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------

with open("model/credit_scoring_model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.main{
    padding-top:20px;
}

.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
}

h1{
    color:#0F52BA;
    font-weight:bold;
}

div[data-testid="stMetric"]{
    background:#F4F8FF;
    padding:15px;
    border-radius:12px;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:10px;
    background:#0F52BA;
    color:white;
    border:none;
}

.stButton>button:hover{
    background:#083D99;
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------

st.title("🏦 Credit Scoring Prediction System")

st.caption(
    "Predict whether a customer is likely to repay a loan using Machine Learning."
)

st.divider()

# -----------------------------
# INPUTS
# -----------------------------

left,right=st.columns(2)

with left:

    age=st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    income=st.number_input(
        "Annual Income",
        min_value=0,
        value=50000
    )

    emp_length=st.number_input(
        "Employment Length (Years)",
        min_value=0.0,
        value=5.0
    )

    loan_amount=st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000
    )

    loan_percent_income=st.slider(
        "Loan Percent Income",
        0.0,
        1.0,
        0.20
    )
    
    
with right:

    home = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "OWN",
            "MORTGAGE",
            "OTHER"
        ]
    )

    intent = st.selectbox(
        "Loan Purpose",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    grade = st.selectbox(
        "Loan Grade",
        [
            "A","B","C","D","E","F","G"
        ]
    )

    interest = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=10.0
    )

    history = st.number_input(
        "Credit History Length",
        min_value=1,
        value=5
    )
    
    previous_default = st.selectbox(
    "Has the applicant defaulted on a previous loan?",
    ["No", "Yes"]
    )
    
st.divider()

predict = st.button(
    "🔍 Predict Credit Risk",
    use_container_width=True
)

# -----------------------------
# FEATURE ENCODING
# -----------------------------

home_own = 1 if home == "OWN" else 0
home_other = 1 if home == "OTHER" else 0
home_rent = 1 if home == "RENT" else 0

intent_education = 1 if intent == "EDUCATION" else 0
intent_home = 1 if intent == "HOMEIMPROVEMENT" else 0
intent_medical = 1 if intent == "MEDICAL" else 0
intent_personal = 1 if intent == "PERSONAL" else 0
intent_venture = 1 if intent == "VENTURE" else 0

grade_map = {
    "A":1,
    "B":2,
    "C":3,
    "D":4,
    "E":5,
    "F":6,
    "G":7
}

grade = grade_map[grade]

default = 1 if default=="Yes" else 0


# -----------------------------
# PREDICTION
# -----------------------------

if predict:

    input_data = np.array([[
        age,
        income,
        emp_length,
        grade,
        loan_amount,
        interest,
        loan_percent_income,
        default,
        history,

        home_other,
        home_own,
        home_rent,

        intent_education,
        intent_home,
        intent_medical,
        intent_personal,
        intent_venture
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    no_risk = probability[0] * 100
    risk = probability[1] * 100

    st.divider()

    st.subheader("📊 Prediction Result")

    st.progress(int(risk))

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Safe Customer",
            f"{no_risk:.2f}%"
        )

    with col2:
        st.metric(
            "Default Risk",
            f"{risk:.2f}%"
        )

    st.write("")

    if prediction == 0:

        st.success(
            f"""
            🟢 **Low Credit Risk**

            The customer is likely to repay the loan.

            **Risk Probability : {risk:.2f}%**
            """
        )

    elif risk < 70:

        st.warning(
            f"""
            🟡 **Medium Credit Risk**

            The customer should be reviewed before loan approval.

            **Risk Probability : {risk:.2f}%**
            """
        )

    else:

        st.error(
            f"""
            🔴 **High Credit Risk**

            High probability of loan default.

            **Risk Probability : {risk:.2f}%**
            """
        )

st.divider()

st.caption(
    "Developed by Vimal Singh | Machine Learning Internship Project | Random Forest Classifier"
)    
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt  #for graphs
from sklearn.cluster import KMeans  #for model
from  streamlit_option_menu import option_menu  # for option menu sidebar

#page configuration
st.set_page_config(page_title="Mobile User Segment",page_icon="📱",layout="wide")

#sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>📱</h1>",unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Mobile User</h2>",unsafe_allow_html=True)
    
    st.caption("Machine Learning Dashboard")
    selected = option_menu(menu_title="Navigation",options=["Dashboard","Dataset","Statistics","Visualization","Prediction"],
                icons=["house","table","bar-chart","graph-up","cpu"],default_index=0)
    st.divider()
    st.success("🟢 Model Ready")

#load data set
df = pd.read_csv("user.csv")

x = df[["Screen_Time","Data_Usage","Recharge_Amount"]]

#KMeans Model
model = KMeans(n_clusters=3,random_state=42,n_init=10)
df["Cluster"] = model.fit_predict(x)

#dashboard
if selected=="Dashboard":
    st.title("📱 Moblie user segment system")
    st.caption("K-Means Clustering | Unsupervised Learning")
    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Total User :",len(df))
    with c2:
        st.metric("Cluster :",3)
    with c3:
        st.metric("Algorithm :","K-Means")
    with c4:
        st.metric("Features :",x.shape[1])
    st.divider()
    
    left,right = st.columns([2,1])
    with left:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20),use_container_width=True,height=350)
    with right:
        st.subheader("Dataset Information")
        st.write("Row :",len(df))
        st.write("Columns :",len(df))
        st.write("Clusters :",3)
        st.metric("Average Screen Time :",f"{df['Screen_Time'].mean():,.1f}Hours")
        st.metric("Average Recharge :",f"₹{df['Recharge_Amount'].mean():,.0f}")

elif selected == "Dataset":
    st.title("📁 Dataset Eplorer")
    st.caption("Data Set Preview")
    st.dataframe(df,use_container_width=True,height=500)

elif selected == "Statistics":
    st.title("📊 Dataset Statistics")
    
    tab1,tab2,tab3 = st.tabs(["Summery","Missing Value","Cluster Count"])
    with tab1:
        st.write(df.describe())
    with tab2:
        st.write(df.isnull().sum())
    with tab3:
        st.bar_chart(df["Cluster"].value_counts()) 

elif selected == "Visualization":
    st.title("📈 Data Visualization")
    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(4,3))
        ax.scatter(df["Screen_Time"],df["Recharge_Amount"],c=df["Cluster"])
        ax.set_xlabel("Screen Time")
        ax.set_ylabel("Recharge Amount")
        ax.set_title("Scatter Plot")
        st.pyplot(fig)
    with col2:
        fig2,ax2 = plt.subplots(figsize=(4,3))
        ax2.pie(df["Cluster"].value_counts(),labels=["Cluster0","Cluster1","Cluster2"],autopct="%1.1f%%")
        ax2.set_title("Cluster Distribution")
        st.pyplot(fig2)

#prediction
elif selected == "Prediction":
    st.title("🎯User Prediction")
    left,right = st.columns(2)
    with left:
        screen = st.slider("Screen Time :",1.0,12.0,5.0,0.5)
        data = st.number_input("Data Usage",1,100,20)
    with right:
        recharge = st.number_input("Recharge Amount :",50,2000,399,step=100)
    st.divider()
    if st.button("Prediction user category",use_container_width=True):
        user = [[screen,data,recharge]]
        cluster = model.predict(user)[0]
        cluster_name = {
            0:"🔵 Light User",
            1:"🟢 Regular User",
            2:"🔥 Heavey User"
        }
        st.success(f"Predict Category :{cluster_name[cluster]}")
        
        col1,col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Summery")
            if cluster == 0:
                st.info("""
                        Reason
                        - Low Screen Time
                        - Low Data
                        - Low Recharge Amount
                        """)
            elif cluster == 1:
                st.info(""")
                        Reason
                        - Low Screen Time
                        - Low Data
                        - Low Recharge Amount
                        """)
            else:
                st.info("""
                        Reason
                        - High Screen Time
                        - High Data
                        - High Recharge Amount
                        """)
        with col2:
            st.subheader("Enter Details")
            st.write(f"**📱 Screen Time :** {screen}Hours")
            st.write(f"**🌐 Data Usage :** {data}GB")
            st.write(f"**💳 Recharge Amount :** ₹{recharge}")
st.caption("Developed By Rajnish Kumar")





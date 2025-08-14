# --- use ngrok to run App ---

# 1. install ngrok
!pip install -q pyngrok

# 2. write App file (if haven't done)
#%%writefile 3_prototype_app.py
# paste prototype_app code

# 3. run ngrok
from pyngrok import ngrok
import os


# copy the Authtoken from your ngork website
AUTHTOKEN = "31BbUcZgPiyIcOxQIeMscRCOa8z_2mNXY1vw5Qe5Drp6QzDUf"

os.system(f"ngrok config add-authtoken {AUTHTOKEN}")

# sun Streamlit App in the background
!streamlit run 3_prototype_app.py &>/dev/null&

# build tunnels and get the public link
public_url = ngrok.connect(8501)
print(f"✅ Your Streamlit app is live at: {public_url}")

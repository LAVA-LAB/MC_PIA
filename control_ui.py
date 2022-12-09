import streamlit as st
from streamlit_autorefresh import st_autorefresh
import subprocess
import os


def execute(cmd):
    popen = subprocess.Popen(cmd, universal_newlines=True, shell=True)


def create_safe_gym_training_command_str(user_input):
    print("================")
    command_parameters = ""
    for key in user_input.keys():
        if key == "prism_dir":
            parameter =  "--"+key+'="'+str(user_input[key])+'"'
        else:
            parameter =  "--"+key+'='+str(user_input[key])+''

        if user_input[key] != "":
            #print("--"+key+"="+str(user_input[key]))
            command_parameters += parameter + " "
    command = "python cool_mc.py " + command_parameters
    print('=======================')
    print(command)
    print('=======================')
    return command
  

st.set_page_config(page_title="COOL-MC", page_icon=":bar_chart:", layout='wide')

css = """

    <style>
    .stButton>button {
margin-top:1.89em;
line-height: 250%;

width: 100%;
}
.big-font {
    font-size:0.8em !important;
    color: white;
}
    </style>
    """
st.markdown(css, unsafe_allow_html=True)

user_input = {}
user_input['task'] = st.sidebar.selectbox(label='Task', options=['safe_training', 'rl_model_checking', 'openai_training'])

if user_input['task'] == 'safe_training':
    st.title("Safe Gym Training")
    st.subheader("Project")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    user_input["project_name"] = col1.text_input("project_name")
    user_input["parent_run_id"] = col2.text_input("Parent ID")
    user_input["prism_file_path"] = col3.text_input("Prism File Path")
    user_input["reward_flag"] = col4.selectbox("Reward Flag", options=["1", "0"])
    user_input["deploy"] = col5.selectbox("Deploy", options=["0", "1"])
    user_input["prism_dir"] = col1.text_input("PRISM Directory", value="../prism_files")
    user_input["wrong_action_penalty"] = col6.text_input("Wrong Action Penalty", value="1000")
    user_input["max_steps"] = col2.text_input("Maximal Steps")
    user_input["constant_definitions"] = col3.text_input("Constant Definitions", value="")
    user_input["sliding_window"] = col4.text_input("Sliding Window", value="100")
    user_input["disable_features"] = col5.text_input("Disable Features", value="")
    st.subheader("RL Agent")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    user_input["rl_algorithm"] = col1.selectbox("Algorithm:",options=['dqn_agent', 'sarsamax'])
    if  user_input["rl_algorithm"] == "dqn_agent":
        user_input["layers"] = col2.selectbox("Layers", options=list(range(0,101)))
        user_input["neurons"] = col3.text_input("Neurons", value="128")
        user_input["replace"] = col4.text_input("Replace", value="1001")
        user_input["replay_buffer_size"] = col5.text_input("Replay Buffer Size", value="100000")
        user_input["batch_size"] = col6.text_input("Batch Size", value="32")

    st.subheader("Learning Parameters")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    number_of_episodes = col1.text_input("Number of Episodes", value = "10000")
    if  user_input["rl_algorithm"] == "dqn_agent":
        user_input["lr"] = col2.text_input("Learning Rate", value="0.0001")
    else:
        user_input["alpha"] = col2.text_input("Alpha", value="0.6")
    user_input["gamma"] = col3.text_input("Gamma", value="0.99")
    user_input["epsilon"] = col4.text_input("Epsilon", value=1)
    user_input["epsilon_decay"] = col5.text_input("Epsilon Decay", value="0.99999")
    user_input["epsilon_min"] = col6.text_input("Epsilon Minimum", value="0.1")
    st.subheader("Model Checking")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    user_input["prop"] = col1.text_input("Property Query", value="0.99")
    user_input["abstraction_input"] = col2.text_input("Abstraction Input", value="")
    user_input["permissive_input"]= col3.text_input("Permissive Input", value="")
    #_ = col2.markdown('<p class="big-font">t</p>', unsafe_allow_html=True)
    start_training = col4.button("Start Training")
    if start_training:
        command_str = create_safe_gym_training_command_str(user_input)
        execute(command_str)
        
    stop_training = col5.button("Stop Training")
        
    if stop_training:
        os.kill(st.session_state['pid'],9)

elif user_input['task'] == 'rl_model_checking':
    st.title("RL Model Checking")
    col1,col2,col3 = st.columns(3)
    user_input["project_name"] = col1.text_input("Project")
    user_input["parent_run_id"] = col2.text_input("Parent ID")
    user_input["env"] = col1.text_input("Environment")
    user_input["prism_dir"] = col3.text_input("PRISM Directory", value="../prism_files")
    user_input["constant_definitions"] = col3.text_input("Constant Definitions", value="")
    user_input["prop"] = col1.text_input("Property Query", value="0.99")
    user_input["abstraction_input"] = col2.text_input("Abstraction Input", value="")
    user_input["permissive_input"]= col2.text_input("Permissive Input", value="")
    start_training = col3.button("Verify")
    if start_training:
        command_str = create_safe_gym_training_command_str(user_input)
        execute(command_str)
        print(command_str)
elif user_input['task'] == 'openai_training':
    st.title("OpenAI Gym Training")
    st.subheader("Project")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    user_input["project_name"] = col1.text_input("project_name")
    user_input["parent_run_id"] = col2.text_input("Parent ID")
    user_input["env"] = col3.text_input("Environment")
    user_input["deploy"] = col5.selectbox("Deploy", options=["0", "1"])
    user_input["sliding_window"] = col4.text_input("Sliding Window", value="100")
    st.subheader("RL Agent")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    user_input["rl_algorithm"] = col1.selectbox("Algorithm:",options=['dqn_agent', 'sarsamax'])
    if  user_input["rl_algorithm"] == "dqn_agent":
        user_input["layers"] = col2.selectbox("Layers", options=list(range(0,101)))
        user_input["neurons"] = col3.text_input("Neurons", value="128")
        user_input["replace"] = col4.text_input("Replace", value="1001")
        user_input["replay_buffer_size"] = col5.text_input("Replay Buffer Size", value="100000")
        user_input["batch_size"] = col6.text_input("Batch Size", value="32")

    st.subheader("Learning Parameters")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    number_of_episodes = col1.text_input("Number of Episodes", value = "10000")
    if  user_input["rl_algorithm"] == "dqn_agent":
        user_input["lr"] = col2.text_input("Learning Rate", value="0.0001")
    else:
        user_input["alpha"] = col2.text_input("Alpha", value="0.6")
    user_input["gamma"] = col3.text_input("Gamma", value="0.99")
    user_input["epsilon"] = col4.text_input("Epsilon", value=1)
    user_input["epsilon_decay"] = col5.text_input("Epsilon Decay", value="0.99999")
    user_input["epsilon_min"] = col6.text_input("Epsilon Minimum", value="0.1")

    #_ = col2.markdown('<p class="big-font">t</p>', unsafe_allow_html=True)
    start_training = col4.button("Start Training")
    if start_training:
        command_str = create_safe_gym_training_command_str(user_input)
        execute(command_str)
        
    stop_training = col5.button("Stop Training")
        
    if stop_training:
        os.kill(st.session_state['pid'],9)



# Timer
count = st_autorefresh(interval=2000, limit=None, key="fizzbuzzcounter")
if count >= 0:
    try:
        f = open('front_end_output.txt','r')
        lines = f.readlines()
        for idx,line in enumerate(lines):
            if line.find("PID: ") != -1:
                pid = line.split(" ")[1]
                st.session_state['pid'] = int(pid)
            st.write(line)
        f.close()
    except:
        st.write("Hi")
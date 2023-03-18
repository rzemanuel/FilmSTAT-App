import json
import cv2
import requests
import pandas as pd
import streamlit as st
import numpy as np
import base64
from PIL import Image
import colour
from scipy.interpolate import interp2d


def macbeth_find(chart_corners, n_x, n_y):
    # create normalized chart between 0-1
    chart_corners1 = np.array(chart_corners).reshape(4, 2).astype(np.int32)
    # chart_corners1[:, 1] = 1080 - chart_corners1[:, 1]
    xs = np.linspace(0, 1, n_x, n_y)
    ys = np.linspace(0, 1, n_y)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.array([xx.flatten(), yy.flatten()]).T
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    fx = interp2d(corners[:, 0], corners[:, 1], chart_corners1[:, 0], kind='linear')
    fy = interp2d(corners[:, 0], corners[:, 1], chart_corners1[:, 1], kind='linear')

    output_list = []
    for i in range(grid.shape[0]):
        output_list.append((fx(grid[i, 0], grid[i, 1]), fy(grid[i, 0], grid[i, 1])))
    output_list = np.array(output_list, dtype=int).reshape(-1, 2)
    return output_list


def sample_image(output_list, img):
    w = [img[output_list[i, 1] - 5:output_list[i, 1] + 5,
         output_list[i, 0] - 5:output_list[i, 0] + 5, :]
         for i in range(output_list.shape[0])]
    samples = np.mean(np.array(w).reshape(output_list.shape[0], 100, 3), axis=1, )
    return samples

headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logOutSection = st.container()
createProjectSection = st.container()
openProjectSection = st.container()
loadFilesSection = st.container()
showImagesSection = st.container()
editChartSection = st.container()
testLUTSection = st.container()

def show_create_project_page():
    mainSection.empty()
    with createProjectSection:
        project_name = st.text_input(label="project_name", value="", key = '234', placeholder="Enter your project name").replace(' ', '_')
        source = st.text_input(label="source", value="", key='23423',placeholder="Enter source camera").replace(' ', '_')
        target = st.text_input(label="target", value="", key='434',placeholder="Enter target camera").replace(' ', '_')
        spectrum = st.text_input(label="spectrum", value="", key='222',placeholder="Enter light spectrum").replace(' ', '_')

        if project_name and source and target and spectrum:
            if st.button("Sumbit", key='submit' ):
                url = 'http://127.0.0.1:8000/projects/'
                headers = {
                    'accept': 'application/json',
                    'Authorization': f'Bearer {st.session_state["token"]}',
                    'Content-Type': 'application/json'}
                data = {
                "name": f'{project_name}',
                "source": f'{source}',
                "target": f"{target}",
                "spectrum": f"{spectrum}"
                    }
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    st.write(f'Project: {response.json()["name"]} created!')

                    st.button("Go Back", key='create_project_go_back', on_click=CancelCreateProject_Clicked)
                else:
                    st.write("Project creation failed")

        st.button("Cancel", key='create_project_cancel', on_click=CancelCreateProject_Clicked)


def CreateProject_Clicked():
    st.session_state['creating_project'] = True

def CancelCreateProject_Clicked():
    createProjectSection.empty()
    st.session_state['creating_project'] = False


def OpenProject_Clicked():
    st.session_state['open_projects'] = True

def transform_query(query):
    new_list = []
    for dictionary in query:
        new_dict = {}
        for key in dictionary.keys():
            if dictionary[key] == "":
                new_dict[key] = False
            elif key in ["IDS_source", "IDS_target","TDS_source", "TDS_target","model", "LUT"] and key:
                new_dict[key] = True
            else:
                new_dict[key] = dictionary[key]
        new_list.append(new_dict)
    return new_list

def CloseOpenProjects_Clicked():
    st.session_state['open_projects'] = False
    st.session_state['loading_files'] = False
    st.session_state['viewing_images'] = False
    st.session_state['project_id'] = 0

def LoadSource_Clicked(project_id):
    st.session_state['project_id'] = project_id
    st.session_state['open_projects'] = False
    st.session_state['viewing_images'] = False
    st.session_state['loading_files'] = True
    st.session_state['project_id'] = project_id


def serialize_bytes(bytes):
    return base64.b64encode(bytes).decode("utf-8")

def ViewImages_Clicked(project_id):
    st.session_state['project_id'] = project_id
    st.session_state['open_projects'] = False
    st.session_state['loading_files'] = False
    st.session_state['viewing_images'] = True
    st.session_state['project_id'] = project_id

def CompareImage_Clicked(project_id):
    st.session_state['project_id'] = project_id
    st.session_state['open_projects'] = False
    st.session_state['example'] = True
    st.session_state['loading_files'] = False
    st.session_state['viewing_images'] = False
    

def show_open_projects_page():
    mainSection.empty()
    st.button("Go Back", key='go_back_from_open_projects', on_click=CloseOpenProjects_Clicked)
    response = get_projects()
    project_list = transform_query(response)
    if project_list == []:
        st.write("You have no projects")
    else:
        with openProjectSection:
            col1, col2= st.columns(2)
            col_list = [col1, col2,]
            for idx, project in enumerate(project_list):
                ind = np.random.randint(1,100000,1)
                col = col_list[idx % 2]
                df = pd.DataFrame(project, index = ['0']).to_dict()
                with col:
                    st.header(f'Project {idx+1}')
                    st.subheader(df['name']['0'])
                    st.text(f"source: {df['source']['0']}")
                    st.text(f"target: {df['target']['0']}")
                    load_source_images = st.button('Load Source Images', key='load_source_images' + str(ind), on_click=LoadSource_Clicked,args = [project['id']])
                    if project['IDS_target'] or project['IDS_source']:
                        view_images = st.button('View Images', key='view_images' + str(ind), on_click=ViewImages_Clicked, args =[project['id']])
                    if (project['TDS_target'] == True) and (project['TDS_source'] == True):
                        lutsize = st.text_input(f"LUT size for {project['name']}")
                        train_model_enabled = bool(lutsize)
                        st.button('Train Model', key='Train Model' + str(ind), on_click=TrainModel_Clicked,
                                                args=[project['id'],lutsize], disabled= not train_model_enabled)
                    if (project['model'] == True) and (project['LUT'] == True):
                        st.button('Download LUT', key='download_lut' + str(ind), on_click=GetLUT_Clicked, args =[project['id']])
                        st.button('Image Compare', key='Image_compare' + str(ind), on_click=CompareImage_Clicked, args=[project['id']])







def GetLUT_Clicked(project_id):

    url = f'http://127.0.0.1:8000/LUT/?project_id={project_id}'
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {st.session_state["token"]}',
        }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        LUT = response.json().get('LUT')
        if LUT:
            # Convert the LUT to a byte string
            LUT_bytes = bytes(LUT, 'utf-8')
            # Create a download link that allows the user to download the LUT file
            st.download_button(
                label="Download LUT",
                data=LUT_bytes,
                file_name="filmstat_LUT.cube",
                mime="text/plain"
            )
        else:
            st.warning("No LUT data found.")
    else:
        st.warning("Error retrieving LUT data.")

def train_model(args):
    project_id, lutsize = args
    url = f'http://127.0.0.1:8000/ml/?lut_size={lutsize}&project_id={project_id}'
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {st.session_state["token"]}',
        'Content-Type': 'application/json'}
    data = ''

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        st.balloons()
    else:
        st.error("failed to train model")


def TrainModel_Clicked(project_id,lutsize):
    st.session_state['project_id'] = project_id
    with st.spinner("Making that LUT!"):
        train_model([project_id, lutsize])




def send_files(files_name_bytes, source_target,):
    # Build the multipart/form-data request.
    url = f'http://127.0.0.1:8000/images/?source={source_target}&project_id={st.session_state["project_id"]}'

    headers = {"Content-Type": "application/json",
               'Authorization': f'Bearer {st.session_state["token"]}',
                'accept': "application/json"}
    data = [{'image_name': file_[0], 'bytes': base64.b64encode(file_[1]).decode("utf-8")} for file_ in files_name_bytes]
    data = json.dumps(data)

    # Make the POST request
    response = requests.post(url, headers=headers, data=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the response data
        response_data = response.json()
        st.success("File(s) uploaded successfully")


    else:
        # Show an error message
        st.error(f"{response.status_code}:  Failed to upload file(s)")






def GoBack_OpenProjects_Clicked():
    st.session_state['load_files'] = False
    st.session_state['open_projects'] = True


def show_load_files_page():

    st.session_state['open_projects'] = False

    openProjectSection.empty()
    mainSection.empty()
    with loadFilesSection:
        st.button("Go Back", key='load_files_go_back', on_click=GoBack_OpenProjects_Clicked)
        source_target = st.selectbox("Select Source or Target", ["source", "target"])
        if source_target == 'source':
            source_target = 'true'
            files_source = st.file_uploader("Upload files", key='file_loader_s', accept_multiple_files=True)
            files_name_bytes = [[file.name, file.getvalue()] for file in files_source]

            if files_source:
                send_files(files_name_bytes, source_target)
        else:
            source_target = 'false'
            files_target = st.file_uploader("Upload files", key='file_loader_t', accept_multiple_files=True)
            files_name_bytes = [[file.name, file.getvalue()] for file in files_target]

            if files_target:
                send_files(files_name_bytes, source_target)
        if st.button('reset image folders', key='reset_image_folder'):
            # Build the multipart/form-data request.
            url = f'http://127.0.0.1:8000/images/?project_id={st.session_state["project_id"]}'

            headers = {
                       'Authorization': f'Bearer {st.session_state["token"]}',
                       'accept': "application/json"}
            response = requests.delete(url, headers=headers,)
            if response.status_code ==200:
                st.success('image folders successfully reset')



def show_main_page():
    with mainSection:
        ind = np.random.randint(1,100000,1)
        st.button("Create Project", key=f'create_project'+ str(ind) , on_click = CreateProject_Clicked)

        st.button("Open Project", key='open_project'+str(ind), on_click=OpenProject_Clicked)


def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False
    st.session_state['creating_project'] = False
    st.session_state['open_projects'] = False
    st.session_state['load_files'] = False
    st.session_state['viewing_images'] = False
    st.session_state['example'] = False



def show_logout_page():
    loginSection.empty()
    with logOutSection:
        st.button("Log Out", key="logout", on_click=LoggedOut_Clicked)


def LoggedIn_Clicked(userName, password):
    try:
        token = verify_credentials(userName, password)
        token = token['access_token']
        if token:
            st.session_state['loggedIn'] = True
            st.session_state['token'] = token
    except:
        st.session_state['loggedIn'] = False;
        st.error("Invalid user name or password")

def show_login_page():
    with loginSection:
        if st.session_state['loggedIn'] == False:
            userName = st.text_input(label="user", value="", placeholder="Enter your user name")
            password = st.text_input(label="pass", value="", placeholder="Enter password", type="password")
            st.button("Login", on_click=LoggedIn_Clicked, args=(userName, password))

# Function to verify the user's credentials against the cloud SQL database
def verify_credentials(username, password):
    url = 'http://127.0.0.1:8000/token'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': '',
        'username': f'{username}',
        'password': f'{password}',
        'scope': '',
        'client_id': '',
        'client_secret': ''
    }
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        login = response.json()
        return login
    else:
        return None

def show_images_page():
    openProjectSection.empty()
    with showImagesSection:
        if st.button("go back", key= "goback_images",):
            st.session_state['viewing_images'] = False
        options = ["Source", "Target"]
        s_t = st.selectbox("Select an option", options)


        images = get_images_cached(s_t == 'Source', st.session_state['project_id'], st.session_state['token'])

        options = np.arange(len(images))
        img_ind = st.selectbox("Select image", options)
        if st.sidebar.button("reset saved coordinates", key= "RESET",):
            st.session_state[f"{s_t}_coords"] = []

        selected_image = images[img_ind]
        if selected_image.dtype != np.float32:
            bitdepth = selected_image.dtype.itemsize * 8
            selected_image = selected_image/2**bitdepth




        height, width, _ = selected_image.shape
        print("selected_image dtype: ", selected_image.dtype)

        col1, col2, col3, col4 = st.columns(4)

        # if not st.checkbox("Upload Coordinates"):
        #     tl_x = col1.slider("Top Left X", 0, width, width // 2 - 200, key='tl_x', step=1)
        #     tl_y = col1.slider("Top Left Y", 0, height, height // 2 - 200, key='tl_y', step=1)
        #
        #     tr_x = col2.slider("Top Right X", 0, width, width // 2 + 200, key='tr_x', step=1)
        #     tr_y = col2.slider("Top Right Y", 0, height, height // 2 - 200, key='tr_y', step=1)
        #
        #     bl_x = col3.slider("Bottom Left X", 0, width, width // 2 - 200, key='bl_x', step=1)
        #     bl_y = col3.slider("Bottom Left Y", 0, height, height // 2 + 200, key='bl_y', step=1)
        #
        #     br_x = col4.slider("Bottom Right X", 0, width, width // 2 + 200, key='br_x', step=1)
        #     br_y = col4.slider("Bottom Right Y", 0, height, height // 2 + 200, key='br_y', step=1)
        #
        # else:
        tl_x = col1.text_input("Top Left X", value=str(width // 2 - 200))
        tl_y = col1.text_input("Top Left Y", value=str(height // 2 - 200))


        tr_x = col2.text_input("Top Right X", value=str(width // 2 + 200))
        tr_y = col2.text_input("Top Right Y", value=str(height // 2 - 200))


        bl_x = col3.text_input("Bottom Left X", value=str(width // 2 - 200))
        bl_y = col3.text_input("Bottom Left Y", value=str(height // 2 + 200))


        br_x = col4.text_input("Bottom Right X", value=str(width // 2 + 200))
        br_y = col4.text_input("Bottom Right Y", value=str(height // 2 + 200))


        tl_x = int(tl_x)
        tl_y = int(tl_y)
        # tl_y = 1080 - tl_y

        tr_x = int(tr_x)
        tr_y = int(tr_y)
        # tr_y = 1080 - tr_y

        bl_x = int(bl_x)
        bl_y = int(bl_y)
        # bl_y = 1080 - bl_y

        br_x = int(br_x)
        br_y = int(br_y)
            # br_y = 1080 - br_y


        tl = (tl_x, tl_y)
        tr = (tr_x, tr_y)
        bl = (bl_x, bl_y)
        br = (br_x, br_y)


        # Create a black image with the same size as the original image
        mask = (np.zeros((height, width, 3), np.float32))

        # Draw the circle on the black image
        cv2.circle(mask, tl, 10, (1, 1, 1), -1)
        cv2.circle(mask, tr, 10, (1, 1, 1), -1)
        cv2.circle(mask, bl, 10, (1, 1, 1), -1)
        cv2.circle(mask, br, 10, (1, 1, 1), -1)
        print("mask dtype: ", mask.dtype)
        # Apply the circle as an overlay to the original image
        overlay = create_overlay(selected_image, (tl_x, tl_y), (tr_x, tr_y), (bl_x, bl_y), (br_x, br_y))


        st.image(np.clip(overlay,0,1), caption=f"Overlay", )
        if len(st.session_state[f"Source_coords"])>0:
            print(st.session_state[f"Source_coords"])
            st.success("Source Chart Coordinates Saved")

        if len(st.session_state[f"Target_coords"]) > 0:
                st.success("Target Chart Coordinates Saved")
                print(st.session_state[f"Target_coords"])

        if len(st.session_state[f"Source_coords"])>0 or len(st.session_state[f"Target_coords"])>0:
            chart_x = st.sidebar.text_input('chart chip height')
            chart_y = st.sidebar.text_input('chart chip width')
            if chart_x and chart_y:
                img = overlay.copy()
                coordys = np.array([tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]).astype(int)
                grid = macbeth_find(coordys, int(chart_x), int(chart_y))
                for i in range(grid.shape[0]):
                    img = cv2.circle(img, tuple(grid[i]), 5, (1,0,0), 5)
                st.image(np.clip(img,0,1))
            if st.sidebar.button('Covert Image Data'):

                with st.spinner(f'Processing Source'):
                    create_tab_data("Source",chart_x,chart_y)
                with st.spinner(f'Processing Target'):
                    create_tab_data("Target",chart_x,chart_y)
            if st.sidebar.button("Save Chart Coordinates"):
                st.session_state[f"{s_t}_coords"] = [tl,tr,bl,br]
                st.experimental_rerun()


        else:
            if st.sidebar.button("Save Chart Coordinates"):
                st.session_state[f"{s_t}_coords"] = [tl,tr,bl,br]
                st.experimental_rerun()

def create_tab_data(s_t,chart_x,chart_y):
    if s_t == "Source":
        source = 'true'
    else:
        source = 'false'
    url = f'http://127.0.0.1:8000/data/?source={source}&chart_x={chart_x}&chart_y={chart_y}&project_id={st.session_state["project_id"]}'
    headers = {"Content-Type": "application/json",
               'Authorization': f'Bearer {st.session_state["token"]}',
                'accept': "application/json"}
    data =  {
      "up_left_x": st.session_state[f"{s_t}_coords"][0][0],
      "up_left_y": st.session_state[f"{s_t}_coords"][0][1],
      "up_right_x": st.session_state[f"{s_t}_coords"][1][0],
      "up_right_y": st.session_state[f"{s_t}_coords"][1][1],
      "bottom_left_x": st.session_state[f"{s_t}_coords"][2][0],
      "bottom_left_y": st.session_state[f"{s_t}_coords"][2][1],
      "bottom_right_x": st.session_state[f"{s_t}_coords"][3][0],
      "bottom_right_y": st.session_state[f"{s_t}_coords"][3][1]
    }
    data = json.dumps(data)

    # Make the POST request
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        # Get the response data
        st.success("Tabular Data uploaded successfully")
    else:
        # Show an error message
        st.error(f"{response.status_code}:  Failed to transform file(s)")


def get_projects():

    url = 'http://127.0.0.1:8000/projects/'
    headers = {
                    'accept': 'application/json',
                    'Authorization': f'Bearer {st.session_state["token"]}'
        }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        projects = response.json()

        return projects
    else:
        print("failed to load projects")


# def base64decode(base64_img):
#     img_bytes = base64.b64decode(base64_img)
#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
#     return img


@st.cache()
def get_images_cached(source, project_id, token,):
    if source:
        s_t = "true"
    else:
        s_t = 'false'
    url = f'http://127.0.0.1:8000/images/images?source={s_t}&project_id={project_id}'
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        images = response.json()
        output=[]
        images_bytes_list = images['files']
        for i in images_bytes_list:
            image_bytes = base64.b64decode(i)
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output.append(img)
        return output

@st.cache(allow_output_mutation=True)
def create_overlay(selected_image, tl, tr, bl, br):
    height, width, _ = selected_image.shape
    # Create a black image with the same size as the original image
    mask = (np.zeros((height, width, 3), np.float32))


    # Draw the circle on the black image
    cv2.circle(mask, tl, 3, (1, 1, 1), -1)
    cv2.circle(mask, tr, 3, (1, 1, 1), -1)
    cv2.circle(mask, bl, 3, (1, 1, 1), -1)
    cv2.circle(mask, br, 3, (1, 1, 1), -1)

    # Apply the circle as an overlay to the original image
    overlay = cv2.addWeighted(selected_image , 0.5, mask, 0.5, 0)
    return overlay


def image_comparison():
    # Add a title
    st.title("Image Comparison")

    # Upload the two images
    left_image = st.file_uploader("Upload left image", key='imageleft', type=["jpg", "jpeg", "png"])
    right_image = st.file_uploader("Upload right image", key='imageright', type=["jpg", "jpeg", "png"])

    # If the images have been uploaded
    if left_image is not None and right_image is not None:
        # Open and display the images
        with st.container():
            col1, col2 = st.columns(2)

            # Open the images using PIL
            left = Image.open(left_image)
            right = Image.open(right_image)
            left_applied = np.array(left)
            bitdepth = left_applied.dtype.itemsize * 8
            left_applied = left_applied/bitdepth
            url = f'http://127.0.0.1:8000/LUT/?project_id={st.session_state["project_id"]}'
            headers = {
                'accept': 'application/json',
                'Authorization': f'Bearer {st.session_state["token"]}',
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                LUT = response.json().get('LUT')
                if LUT:
                    # Convert the LUT to a byte string

                    LUT_bytes = bytes(LUT, 'utf-8')
                    LUT_str = LUT_bytes.decode('utf-8')
                    size = int(LUT_str.split('\n')[0].split()[1])
                    lines = LUT_str.split('\n')[1:]
                    print(len(lines))
                    print()
                    print()
                    print(lines[0])
                    # Split each line into elements and convert to float
                    data = [[float(elem) for elem in line.split()] for line in lines]
                    shape = left_applied.shape
                    # Create a numpy array from the data
                    LUT_array = np.array(data[:-1])
                    lut = colour.LUT3D(LUT_array.reshape(size, size, size, 3))
                    left_applied = lut.apply(left_applied.reshape(-1,3))
                    left_applied = left_applied.reshape(shape)
                    left_applied_ = np.clip(left_applied, 0,1)
                    print(left_applied.max())



            # Display the images side by side
            with col1:
                st.image(left, use_column_width=True, caption="Left image")
                st.image(left_applied_, use_column_width=True, caption="Left image w/ LUT")
            with col2:
                st.image(right, use_column_width=True, caption="Right image")





def test_lut_page():
    image_comparison()





with headerSection:
    st.title("filmSTAT AI Camera Match")

    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False
        show_login_page()
    else:
        if st.session_state['loggedIn']:
            show_logout_page()
            if 'creating_project' not in st.session_state:
                st.session_state['creating_project'] = False
                if 'open_projects' not in st.session_state:
                    st.session_state['open_projects'] = False
                    if 'project_id' not in st.session_state:
                        st.session_state['project_id'] = None
                        if 'loading_files' not in st.session_state:
                            st.session_state['loading_files'] = False
                            if 'viewing_images' not in st.session_state:
                                st.session_state['viewing_images'] = False
                                if 'chart' not in st.session_state:
                                    st.session_state['chart'] = False
                                    if 'Source_coords' not in st.session_state:
                                        st.session_state[f"Source_coords"] = []
                                        if 'Target_coords' not in st.session_state:
                                            st.session_state[f"Target_coords"] = []
                                            if 'training_model' not in st.session_state:
                                                st.session_state['training_model'] = False
                                                if "example" not in st.session_state:
                                                    st.session_state['example'] = False
                                                    show_main_page()


            else:
                if st.session_state['creating_project']:
                    show_create_project_page()
                else:
                    if st.session_state['open_projects']:
                        show_open_projects_page()
                    else:
                        if st.session_state['loading_files']:
                            show_load_files_page()
                        elif st.session_state['viewing_images']:
                            show_images_page()
                        elif st.session_state['example']:
                            test_lut_page()
                        else:
                           show_main_page()



        else:
            show_login_page()




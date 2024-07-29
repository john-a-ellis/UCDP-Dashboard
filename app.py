#import dependencies
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc
import math
# import dash_ag_grid as dag
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template
# from langchain_huggingface import HuggingFacePipeline 
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
# from huggingface_hub import login
# from assets.api_keys import huggingface_key


# login(token = huggingface_key)

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template("darkly")


# supporting UDF's

# def summarize_conflict(df):
    
#     # Initialize the model and tokenizer
#     model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # Load the model configuration
#     config = AutoConfig.from_pretrained(model_name)

#     # Modify the rope_scaling parameter
#     # config.rope_scaling = {'type': 'llama3', 'factor': 8.0}

#     # Load the model with the modified configuration
#     model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

#     # Create the pipeline
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=500
#     )

#     # Create the LangChain LLM
#     llm = HuggingFacePipeline(pipeline=pipe)

#     # Prepare the input for the model
#     conflict_name = df['conflict_name'].iloc[0]
#     events = df.to_dict('records')
    
#     prompt = f"""Summarize the conflict described by the following events:

#     Conflict Name: {conflict_name}

#     Events:
#     {events}

#     Please provide a concise summary of the conflict, including its nature, timeline, and impact on civilians.
#     """

#     # Generate the summary
#     summary = llm(prompt)

#     return summary

def create_columnDefs(df):
    columnDefs = []
    for col in df.columns:
        column_def = {
            "field": col,
            "headerName": col.replace('_', ' ').title(),
            "filter": True,
            "sortable": True
        }
        
        # Determine the column type and set appropriate properties
        if df[col].dtype == 'object':
            column_def["filter"] = "agTextColumnFilter"
        elif df[col].dtype in ['int64', 'float64']:
            column_def["filter"] = "agNumberColumnFilter"
            column_def["type"] = "numericColumn"
        elif df[col].dtype == 'bool':
            column_def["filter"] = "agSetColumnFilter"
            column_def["cellRenderer"] = "agBooleanCellRenderer"
        elif df[col].dtype in ['datetime64[ns]','timedelta64[ns]']:
            column_def["filter"] = "agDateColumnFilter"
            column_def["type"] = "dateColumn"
        
        columnDefs.append(column_def)
    
    return columnDefs

def geographic_centroid(coordinates):
    """
    Calculate the geographic centroid of a set of lat/lon coordinates.
    
    :param coordinates: List of [latitude, longitude] pairs
    :return: (latitude, longitude) of the centroid
    """
    # Convert lat/lon to Cartesian coordinates
    x, y, z = 0, 0, 0
    for lat, lon in coordinates:
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x += math.cos(lat_rad) * math.cos(lon_rad)
        y += math.cos(lat_rad) * math.sin(lon_rad)
        z += math.sin(lat_rad)

    # Calculate average
    total = len(coordinates)
    if total == 0:
        total = 1

    x /= total
    y /= total
    z /= total

    # Convert average Cartesian coordinates back to lat/lon
    central_lon = math.atan2(y, x)
    central_sqrt = math.sqrt(x * x + y * y)
    central_lat = math.atan2(z, central_sqrt)

    return math.degrees(central_lat), math.degrees(central_lon)





# import data set and transform as necessary
df=pd.read_csv('data/GEDEvent_v24_1.csv')

#add a duration feature
df['duration']=pd.to_datetime(df['date_end'])-pd.to_datetime(df['date_start'] )

#create lists for various controls
select_by_dict = {
    1: "Conflict Name",
    2: "Combatant Side A",
    3: "Combatant Side B",
    4: "Country"
}


myCountry = df['country'].unique().tolist()
myCountry.insert(0,"All")
myConflicts_df = df["conflict_name"].sort_values().drop_duplicates()
myConflicts = myConflicts_df.to_list()
myConflicts.insert(0, 'All')

mySideA_df = df["side_a"].sort_values().drop_duplicates()
mySideA = mySideA_df.to_list()
mySideA.insert(0,'All')

mySideB_df = df["side_b"].sort_values().drop_duplicates()
mySideB = mySideB_df.to_list()
mySideB.insert(0,'All')


#create some initial components
df1=df.copy()
fig = px.density_mapbox(df1.sort_values('date_start'), lat='latitude', lon='longitude', z='best', radius=20,
                        center=dict(lat=0, lon=28), zoom=1, 
                        mapbox_style="carto-positron", 
                        height= 400,
                        width=1250,
                        title = 'All Events',
                        # range_color=[1,100000],
                        hover_data = ['id', 'conflict_new_id', 'conflict_name'])
fig['data'][0]['colorbar']['title']['text'] = 'Deaths'
fig['data'][0].update(hovertemplate =("ID: %{customdata[0]}<br>"
                                        "Conflict ID: %{customdata[1]}<br>"
                                        "Latitude: %{lat}<br>"
                                        "Longitude: %{lon}<br>"
                                        "Estimated Deaths: %{z}<br>"
                                        "Name: %{customdata[2]}<br>"
                                        "<extra></extra>"
                                        ),)
fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), coloraxis_colorbar_title_text='Est. Deaths')

default_map = go.Figure(fig)

fig1 = px.area(df1.sort_values('date_start'), 
               x='date_start', 
               y = ['best', 'deaths_civilians'], 
               title='All Events', 
               height=400)
# fig1.add_trace(trendline=dict(
#         type='ols',  
#         color='red',
#         dash='dash'))
fig1.data[0].name = 'Best Estimate'
fig1.data[1].name = 'Civilians Only'
fig1.update_layout(legend_title_text = "Deaths",
                    xaxis_title=None,
                    yaxis_title=None,
                    margin=dict(l=10, r=10, t=40, b=10),
                    )
fig2 = px.scatter(df1.sort_values('year'), 
                  x='date_start', 
                  y='best', 
                  title='All Events', 
                  size='best', 
                  color='region', 
                  height=400)

fig2.update_layout(legend_title_text = "Deaths",
                    xaxis_title=None,
                    yaxis_title=None,
                    margin=dict(l=10, r=10, t=40, b=10)
)
#start the Server
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css])
server = app.server
myTitle = 'The State of Conflict'
app.title = myTitle

#create the layout

app.layout = dbc.Container(
    [
        dbc.Row(
            
                [
                    dbc.Col(
                        html.A(
                            href='https://www.linkedin.com/in/john-a-ellis/', 
                            target = '_blank', 
                            children=html.Img(
                                src='assets/favicon.ico', 
                                height=80, 
                                width=80, 
                                id='cogent-logo',  
                                alt='Cogent Analytics') 
                                ), 
                                width = 1,),
                    
                    dbc.Col(html.H1(myTitle, className="text-center text-white bg-primary rounded mb-4"), width = 10 ),
                ], 
        ),

        dbc.Row(
                [
                    dbc.Col(dbc.RadioItems(select_by_dict, value='1', id='select_by_type',inline=True, switch=True,  className="ms-5"), width= 2  ),
                    dbc.Col([
                                dbc.Label('Conflict Name', id='select_label'),
                                dbc.Select(options=myConflicts, value='All', id='my-selection', ),
                                
                             ], width=4),
                ], 
                ),
        dbc.Row( [ 
            html.Hr( style={'borderColor':'white' }),
                dbc.Col(
                    [dbc.Tabs([
                        dbc.Tab(
                            dbc.Card(
                                dbc.CardBody([
                                    # html.Div('World View'),
                                    dcc.Graph(id='map_fig', figure = fig),
                                    html.P('Event Details:', style={'height':'20px', 'width':'1250px'}, className = "card-title"),
                                    html.Pre('', style={'height':'150px', 'width':'1250px', 'overflow-y':'auto'},id='event-details',),
                                ],  className="bg-secondary text-light",)
                            ), label='World View'
                        ),
                        dbc.Tab(
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(id = 'line_fig', figure = fig1),
                                    html.P('Conflict Summary:', style={'height':'20px', 'width':'1250px'}, className = "card-title"),
                                    html.Pre('', style={'height':'150px', 'width':'1250px', 'overflow-y':'auto'},id='conflict-summary',),
                                ],    className="bg-secondary text-light",)
                                
                            ), label='Confict over Time'
                        ),
                         dbc.Tab(
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(id = 'bubble_fig', figure = fig2),
                                ])
                            ), label='Country Involvement'   
                        ),
                    ]
                      )  
                    ], 
                        width=12, align="stretch",
                    ),
                    ]),
    ],
    
    
# fluid = True,
class_name='dashboard-container border_rounded',

# style={'display':'flex'}
)

@app.callback(
        Output(component_id='map_fig', component_property='figure', allow_duplicate=True),
        Output(component_id='line_fig', component_property='figure', allow_duplicate=True),
        Output(component_id='bubble_fig', component_property='figure', allow_duplicate=True),
        Input(component_id ='my-selection', component_property='value'),
        Input(component_id ='select_label', component_property='children'),  
        prevent_initial_call='initial_duplicate',
        suppress_callack_exceptions=True
)
def select_conflict(thisSelection, thisLabel):
    # print (f'Label: {thisLabel}')
    print(f'Selection: {thisSelection}')
    if thisLabel == 'Conflict Name':
        findit = 'conflict_name'
    elif thisLabel == 'Combatant Side A':
        findit = 'side_a'
    elif thisLabel == 'Combatant Side B':
        findit = 'side_b'
    elif thisLabel == 'Country':
        findit ='country'
    print(f'findit: {findit}')
    myChartTitle = thisLabel + " - " + thisSelection

    if thisSelection != "All":
        df1=df[df[findit]==thisSelection]
    else:
        df1=df
    
    coordinates=df1[['latitude', 'longitude']].to_dict(orient='split')
    myLat, myLon = geographic_centroid(coordinates['data'])
    # print(myLat, myLon)
    fig['data'][0].update(lat=df1['latitude'], lon=df1['longitude'], z=df1['best'],
                    hovertemplate =("ID: %{customdata[0]}<br>"
                                    "Conflict ID: %{customdata[1]}<br>"
                                    "Latitude: %{lat}<br>"
                                    "Longitude: %{lon}<br>"
                                    "Estimated Deaths: %{z}<br>"
                                        "Name: %{customdata[2]}<br>"
                                    "<extra></extra>"
                                    ),
                                    customdata=df1[['id', 'conflict_new_id', 'conflict_name']].values
    )
    fig.update_layout(mapbox=dict(center=dict(lat=myLat, lon=myLon),
                                              zoom=1),
                                              title=myChartTitle
                                              )
    
    fig1['data'][0].update(x = df1['date_start'].sort_values(), y = df1['best'])
    if len(fig1['data']) < 2:
        fig1.add_trace(go.Scatter(x = df1['date_start'].sort_values(), y = df1['deaths_civilians'], mode = ' '))
    else:
        fig1['data'][1].update(x = df1['date_start'].sort_values(), y = df1['deaths_civilians'])

    fig1.update_layout(title =  myChartTitle)

  

    fig2 = px.scatter(df1.sort_values('year'), 
                  x='year', 
                  y='best', 
                  size='best', 
                  color='country', 
                  height=400)
    
    fig2.update_layout(legend_title_text = "Deaths",
                    xaxis_title=None,
                    yaxis_title=None,
                    margin=dict(l=10, r=10, t=40, b=10),
                    title =  myChartTitle)
    
    return fig, fig1, fig2

@app.callback(
    Output(component_id='select_label', component_property='children'),
    Output(component_id='my-selection', component_property='options'),
    Output(component_id = 'event-details', component_property='children', allow_duplicate=True),
    Input(component_id='select_by_type', component_property='value'),
    prevent_initial_call=True,
    # suppress_callback_exceptions=True,
)

def update_select(selectType):

    mySelection = select_by_dict[int(selectType)]
    myDiv = " "
    if selectType == '1':
        myOptions = myConflicts
        # myValues = myConflicts['value']
    elif selectType == '2':
        myOptions = mySideA
        # myValues = mySideA['side_a_new_id']
    elif selectType == '3':
        myOptions = mySideB
        # myValues = mySideB['side_b_new_id']    
    elif selectType == '4':
        myOptions = myCountry

    return mySelection, myOptions, myDiv

@app.callback(
    Output(component_id='event-details', component_property='children'),
    # Output(component_id='conflict-summary', component_property='children'),
    Output(component_id = 'map_fig', component_property ='figure', allow_duplicate=True),
    Output(component_id = 'line_fig', component_property='figure'),
    Output(component_id = 'bubble_fig', component_property='figure'),
    Input(component_id='map_fig', component_property='clickData'),
    prevent_initial_call=True,
    suppress_callback_exceptions=True,
   
)
def update_div(data_clicked):

    dff = df1[df1['id'] == data_clicked['points'][0]['customdata'][0]][['country', 'conflict_name', 'where_description', 'source_article','source_headline' ]]
    df2 = df1[df1['conflict_new_id'] == data_clicked['points'][0]['customdata'][1]][['date_start', 'source_article', 'source_headline', 'best', 'deaths_civilians', 'conflict_name', 'country', 'year']]
    df3 = df1[df1['conflict_new_id']== data_clicked['points'][0]['customdata'][1]]
 
    myDiv = ""
    # myConflictSummary = ""
    # myConflictSummary = summarize_conflict(dff)
    myChartTitle = dff['conflict_name'].values[0]
    for this in dff.columns.tolist():
       myDiv += f"{this.replace('_', ' ').title()}: {dff[this].values[0]}\n"
  
    fig1['data'][0].update(x = df2['date_start'].sort_values(), y = df2['best'])

    if len(fig1['data']) < 2:
        fig1.add_trace(go.Scatter(x = df2['date_start'].sort_values(), y = df2['deaths_civilians'], mode=''))
    else:
        fig1['data'][1].update(x = df2['date_start'].sort_values(), y = df2['deaths_civilians'])

    fig1.update_layout(title = myChartTitle)
    
    
    fig2 = px.scatter(df2.sort_values('year'), 
                  x='year', 
                  y='best', 
                #   title=myCountry, 
                  size='best', 
                  color='country', 
                  height=400)
    
    fig2.update_layout(legend_title_text = "Deaths",
                    xaxis_title=None,
                    yaxis_title=None,
                    margin=dict(l=10, r=10, t=40, b=10),
                    title = myChartTitle)
    
    fig['data'][0].update(lat=df3['latitude'], lon=df3['longitude'], z=df3['best'],
                        hovertemplate =("ID: %{customdata[0]}<br>"
                                        "Conflict ID: %{customdata[1]}<br>"
                                        "Latitude: %{lat}<br>"
                                        "Longitude: %{lon}<br>"
                                        "Estimated Deaths: %{z}<br>"
                                         "Name: %{customdata[2]}<br>"
                                        "<extra></extra>"
                                        ),
                                        customdata=df3[['id', 'conflict_new_id', 'conflict_name']].values)
    
    fig.update_layout(mapbox=dict(center=dict(lat=data_clicked['points'][0]['lat'],
                                              lon=data_clicked['points'][0]['lon']),
                                              zoom=4),
                                              title = myChartTitle
                                              )
    
    return myDiv, fig, fig1, fig2,

if __name__ == '__main__':
    app.run(jupyter_mode='_none', debug=True, port=3050)
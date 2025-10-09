import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import warnings
warnings.filterwarnings("ignore")

# ------------------ Data Loading ------------------
df = pd.read_csv("data.csv") 
df = df.dropna()

def convert_to_number(x):
    if isinstance(x, str):
        x = x.replace(',', '').replace('K', 'e3').replace('M', 'e6').replace('B', 'e9')
        try:
            return float(eval(x))
        except:
            return np.nan
    return x

df['Followers'] = df['Followers'].apply(convert_to_number)
df['Avg. Likes'] = df['Avg. Likes'].apply(convert_to_number)
df['Posts'] = df['Posts'].apply(convert_to_number)
df['Eng Rate'] = df['Eng Rate'].str.replace('%', '').astype(float)
df = df.dropna()
df.reset_index(drop=True, inplace=True)

# Encode Category
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

# Regression Features
features = ['Followers', 'Avg. Likes', 'Posts', 'Category_encoded']
X = df[features]
y = df['Eng Rate']

# Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
reg_model = RandomForestRegressor()
reg_model.fit(X_train_scaled, y_train)
y_pred_reg = reg_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))

# Classification Model
df['Engagement_Label'] = pd.qcut(df['Eng Rate'], q=2, labels=['Low', 'High'])
y_class = df['Engagement_Label']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
clf_model = RandomForestClassifier()
clf_model.fit(X_train_c, y_train_c)
y_pred_clf = clf_model.predict(X_test_c)
clf_report = classification_report(y_test_c, y_pred_clf)
conf_matrix = confusion_matrix(y_test_c, y_pred_clf)

# ------------------ Dash App ------------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Social Media Analytics Dashboard", style={'textAlign': 'center'}),
    
    html.H3("1️⃣ Model Toggle"),
    dcc.RadioItems(
        id='model_toggle',
        options=[
            {'label': 'Regression (Eng Rate)', 'value': 'regression'},
            {'label': 'Classification (Low/High)', 'value': 'classification'}
        ],
        value='regression',
        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
    ),
    
    html.H3("2️⃣ Data Filters"),
    html.Div([
        html.Label("Followers Range:"),
        dcc.RangeSlider(
            id='followers_slider',
            min=df['Followers'].min(),
            max=df['Followers'].max(),
            step=1000,
            marks={int(df['Followers'].min()): str(int(df['Followers'].min())),
                   int(df['Followers'].max()): str(int(df['Followers'].max()))},
            value=[df['Followers'].min(), df['Followers'].max()]
        ),
        html.Label("Engagement Rate Range:"),
        dcc.RangeSlider(
            id='eng_slider',
            min=df['Eng Rate'].min(),
            max=df['Eng Rate'].max(),
            step=0.1,
            marks={int(df['Eng Rate'].min()): str(int(df['Eng Rate'].min())),
                      int(df['Eng Rate'].max()): str(int(df['Eng Rate'].max()))},
       value=[df['Eng Rate'].min(), df['Eng Rate'].max()]
        ),
        html.Label("Number of Posts:"),
        dcc.RangeSlider(
            id='posts_slider',
            min=df['Posts'].min(),
            max=df['Posts'].max(),
            step=1,
            marks={int(df['Posts'].min()): str(int(df['Posts'].min())),
                   int(df['Posts'].max()): str(int(df['Posts'].max()))},
            value=[df['Posts'].min(), df['Posts'].max()]
        ),
    ], style={'margin-bottom': '30px'}),
    
    html.H3("3️⃣ Average Engagement Rate by Category"),
    dcc.Graph(id='eng_rate_bar'),
    
    html.H3("4️⃣ Top Influencers Table"),
    html.Div([
        html.Label("Sort By:"),
        dcc.Dropdown(
            id='top_sort_dropdown',
            options=[
                {'label': 'Followers', 'value': 'Followers'},
                {'label': 'Engagement Rate', 'value': 'Eng Rate'},
                {'label': 'Average Likes', 'value': 'Avg. Likes'}
            ],
            value='Followers'
        ), 
     dash_table.DataTable(
            id='top_table',
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
            sort_action='native',
            style_table={'overflowX': 'auto'}
        )
    ]),
    
    html.H3("5️⃣ Scatter & Boxplots"),
    html.Div([
        html.Label("Select Category:"),
        dcc.Dropdown(
            id='category_dropdown',
            options=[{'label': cat, 'value': cat} for cat in df['Category'].unique()],
            value=df['Category'].unique()[0]
        ),
        dcc.Graph(id='followers_likes_scatter'),
        dcc.Graph(id='category_boxplot')
    ]),
    
    html.H3("6️⃣ Interactive Prediction"),
    html.Div([
        html.Label("Followers:"), dcc.Input(id='input_followers', type='number', value=1000),
        html.Label("Average Likes:"), dcc.Input(id='input_likes', type='number', value=100),
        html.Label("Posts:"), dcc.Input(id='input_posts', type='number', value=10),
        html.Label("Category:"), dcc.Dropdown(id='input_category',
                                             options=[{'label': c, 'value': c} for c in df['Category'].unique()],
                                             value=df['Category'].unique()[0]),
        html.Button("Predict", id='predict_btn', n_clicks=0),
        html.Div(id='prediction_output', style={'margin-top': '20px', 'fontWeight': 'bold'})
    ]),

dcc.Graph(id='model_perf')
])

# ------------------ Callbacks ------------------
@app.callback(
    Output('followers_likes_scatter', 'figure'),
    Output('category_boxplot', 'figure'),
    Output('eng_rate_bar', 'figure'),
    Output('top_table', 'data'),
    Output('prediction_output', 'children'),
    Output('model_perf', 'figure'),
    Input('category_dropdown', 'value'),
    Input('top_sort_dropdown', 'value'),
    Input('followers_slider', 'value'),
    Input('eng_slider', 'value'),
    Input('posts_slider', 'value'),
    Input('predict_btn', 'n_clicks'),
    Input('input_followers', 'value'),
    Input('input_likes', 'value'),
    Input('input_posts', 'value'),
    Input('input_category', 'value'),
    Input('model_toggle', 'value')
)
def update_dashboard(selected_category, top_sort,
                     followers_range, eng_range, posts_range,
                     n_clicks, in_followers, in_likes, in_posts, in_category,
                     model_toggle):
 # Filter data
    filtered_df = df[
        (df['Followers'] >= followers_range[0]) & (df['Followers'] <= followers_range[1]) &
        (df['Eng Rate'] >= eng_range[0]) & (df['Eng Rate'] <= eng_range[1]) &
        (df['Posts'] >= posts_range[0]) & (df['Posts'] <= posts_range[1])
    ]
    
    # Scatter Plot
    scatter_fig = px.scatter(
        filtered_df[filtered_df['Category']==selected_category],
        x='Followers', y='Avg. Likes', hover_data=['name'],
        title=f"Followers vs Likes for {selected_category}"
    )
    
    # Boxplot
    boxplot_fig = px.box(
        filtered_df, x='Category', y='Eng Rate', title="Engagement Rate by Category"
    )
    
    # Average Engagement Bar
    eng_bar = filtered_df.groupby('Category')['Eng Rate'].mean().reset_index()
    bar_fig = px.bar(eng_bar, x='Category', y='Eng Rate', title="Average Engagement Rate by Category")
    
    # Top Influencers Table
    top_df = filtered_df.sort_values(by=top_sort, ascending=False).head(10)
    table_data = top_df.to_dict('records')
 # Interactive Prediction
    input_df = pd.DataFrame({
        'Followers': [in_followers],
        'Avg. Likes': [in_likes],
        'Posts': [in_posts],
        'Category_encoded': [le.transform([in_category])[0]]
    })
    input_scaled = scaler.transform(input_df)
    if n_clicks > 0:
        if model_toggle == 'regression':
            pred = reg_model.predict(input_scaled)[0]
            pred_text = f"Predicted Engagement Rate: {pred:.2f}%"
        else:
            pred_label = clf_model.predict(input_df)[0]
            pred_text = f"Predicted Engagement Class: {pred_label}"
    else:
        pred_text = ""
    
    # Model Performance Visualization
    if model_toggle == 'regression':
        perf_fig = px.scatter(x=y_test, y=y_pred_reg,
                              labels={'x':'Actual', 'y':'Predicted'},
                              title=f"Regression: Actual vs Predicted (RMSE={rmse:.2f})")
    else:
        z = conf_matrix
        perf_fig = ff.create_annotated_heatmap(z, x=['Low','High'], y=['Low','High'],
                                               colorscale='Viridis', showscale=True)
    
    return scatter_fig, boxplot_fig, bar_fig, table_data, pred_text, perf_fig
# ------------------ Run ------------------
if __name__ == '__main__':
    app.run(debug=True)



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("data.csv") 
df.head()
print("Missing values:\n", df.isnull().sum())

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
df.info()
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Followers', y='Avg. Likes', hue='Category')
plt.title("Followers vs Avg. Likes")
plt.show()

# Engagement Rate by Category
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='Category', y='Eng Rate')
plt.xticks(rotation=90)
plt.title("Engagement Rate by Category")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[['Followers', 'Avg. Likes', 'Posts', 'Eng Rate']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

features = ['Followers', 'Avg. Likes', 'Posts', 'Category_encoded']
X = df[features]
y = df['Eng Rate']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
df['Engagement_Label'] = pd.qcut(df['Eng Rate'], q=2, labels=['Low', 'High'])

y_class = df['Engagement_Label']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

model_c = RandomForestClassifier()
model_c.fit(X_train_c, y_train_c)
y_pred_c = model_c.predict(X_test_c)

print(classification_report(y_test_c, y_pred_c))
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Social Media Analytics Dashboard", style={'textAlign': 'center'}),
    
    dcc.Dropdown(
        id='category_dropdown',
        options=[{'label': cat, 'value': cat} for cat in df['Category'].unique()],
        value=df['Category'].unique()[0]
    ),
    
    dcc.Graph(id='followers_likes_scatter'),
    
    dcc.Graph(id='category_boxplot')
])

@app.callback(
    Output('followers_likes_scatter', 'figure'),
    Output('category_boxplot', 'figure'),
    Input('category_dropdown', 'value')
)
def update_graphs(selected_category):
    filtered_df = df[df['Category'] == selected_category]

    scatter_fig = px.scatter(
        filtered_df, x='Followers', y='Avg. Likes',
        title=f"Followers vs Likes for {selected_category}",
        hover_data=['name']
    )
    boxplot_fig = px.box(
        df, x='Category', y='Eng Rate',
        title="Engagement Rate by Category"
    )

    return scatter_fig, boxplot_fig
# 8. Run the dashboard locally
if __name__ == '__main__':
    app.run(debug=True)


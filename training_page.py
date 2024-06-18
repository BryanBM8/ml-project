import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR 
from streamlit import session_state as state
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
import pickle

model_options = {"Random Forest": RandomForestRegressor, "Gradient Boosting": GradientBoostingRegressor, "Support Vector Machine": SVR}

param_options = {
    'n_estimators': [1,1000,100],
    'max_depth': [1,100,3],
    'learning_rate': [0.01,1.0,0.1],
    'min_samples_split': [2,100,2],
    'min_samples_leaf': [1,100,1],
    'degree': [1,10,3],
    'tol': [0.0001,0.01,0.001],
    'C': [0.1,10.0,1.0]
}

param_options_grid_search = {
    'n_estimators': [100,200,300],
    'max_depth': [3,5,7],
    'learning_rate': [0.1,0.2,0.3],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4],
    'degree':[2,3,4],
    'tol': [0.001,0.005,0.01],
    'C': [1,2,3]
}

def training():
    params = {}

    st.title("Training")
    if 'preprocessing_done' not in st.session_state or st.session_state.preprocessing_done == False:
        st.warning("Please complete the preprocessing step first.")
        return
    model_option = st.selectbox("Model:", list(model_options.keys()))
    use_grid_cv = st.toggle('Use Grid Search CV', False)
    params['test_size'] = st.slider("Test Dataset Size", 0.05, 0.9, 0.2) 
    params['random_state'] = st.number_input("random_state", 0,100,42)
    
    if not use_grid_cv:
        # filter params if it's exist in the model or not
        params = dict(params,**dict([(k,st.number_input(k,min_value=r[0],max_value=r[1],value=r[2])) for k,r in param_options.items() 
                       if k in model_options[model_option].__init__.__code__.co_varnames]))
    else:
        params['cv'] = st.number_input("cv", 2,10,5)
        filtered_params = {k: v for k, v in param_options_grid_search.items() if k in model_options[model_option].__init__.__code__.co_varnames}
        params_df = pd.DataFrame.from_dict(
            filtered_params
        )
        edited_param_df = st.data_editor(params_df,width=1440, key='grid_cv_params', num_rows='dynamic')
    print(params)
    
    if st.button("Train"):
        state.training_done = False
        model_params = model_options[model_option].__init__.__code__.co_varnames
        state.x_train, state.x_test, state.y_train, state.y_test = train_test_split(state.df.drop('G3',axis=1), state.df['G3'], test_size=params['test_size'], random_state=params['random_state'])
        if not use_grid_cv:
            model_params = {k: v for k, v in params.items() if k in model_params}
            state.model = model_options[model_option](
                **model_params
            ).fit(state.x_train, state.y_train)
            st.write("accuracy: ", state.model.score(state.x_test, state.y_test))
            st.success("Training Done")
            state.training_done = True
        else:
            if(edited_param_df.isna().sum().sum() > 0):
                st.error("Please fill all the parameters")
            else:
                grid_search_param = dict([(k, edited_param_df[k].tolist()) for k in edited_param_df.columns])
                grid_search = GridSearchCV(estimator=model_options[model_option](**{k: v for k, v in params.items() if k in model_params and k == 'random_state'})
                    , param_grid=grid_search_param, cv=params['cv'], n_jobs=-1, verbose=2)
                with st.spinner('Training...'):
                    grid_search.fit(state.df.drop('G3',axis=1), state.df['G3'])
                st.success("Training Done")
                st.write(f"Best parameters found: {grid_search.best_params_}")
                state.training_done = True
                state.model = grid_search.best_estimator_

            if state.training_done == True:
                with open('model.pkl', 'wb') as mo:
                    pickle.dump(state.model, mo)
                st.session_state.model = state.model
                st.write("accuracy: ", state.model.score(state.x_test, state.y_test))
            


    return
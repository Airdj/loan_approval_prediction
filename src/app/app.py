import gradio as gr
import pandas as pd
from src.pipelines.inference_pipeline import inference_pipeline
cols_path = 'src/utils/columns.csv'
columns = pd.read_csv(cols_path, header=None)[0].tolist()

def predict(*values):
    data = dict(zip(columns, values))
    df = pd.DataFrame([data])
    return inference_pipeline(df)

demo = gr.Interface(
    fn=predict,

    inputs=[

        gr.Number(label="id"),

        gr.Number(label="person_age"),
        gr.Number(label="person_income"),

        gr.Dropdown(['RENT','MORTGAGE','OWN','OTHER'], label="person_home_ownership"),

        gr.Number(label="person_emp_length"),

        gr.Dropdown(['EDUCATION','MEDICAL','VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMEIMPROVEMENT'],
                    label="loan_intent"),

        gr.Dropdown(['A','B','C','D','E','F','G'], label="loan_grade"),

        gr.Number(label="loan_amnt"),

        gr.Number(label="loan_int_rate"),

        gr.Number(label="loan_percent_income"),

        gr.Dropdown(['N', 'Y'], label='cb_person_default_on_file'),

        gr.Number(label="cb_person_cred_hist_length")
        ],

        outputs='text',
        title='Loan Prediction'
)

demo.launch(server_name="0.0.0.0", server_port=7860)
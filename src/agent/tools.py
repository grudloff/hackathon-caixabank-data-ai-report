import pandas as pd

from langchain_core.tools import InjectedToolArg, tool
from typing_extensions import Annotated
from src.data.data_functions import earnings_and_expenses, expenses_summary, cash_flow_summary, get_transactions_dataset

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

import logging

@tool(return_direct=True)
def generate_report(df: Annotated[pd.DataFrame, InjectedToolArg], client_id: Annotated[int, InjectedToolArg], start_date: str, end_date: str) -> dict:
    """
    Generate a report for a client.

    Args:
        df: The DataFrame with the data. (Optional)
        client_id: The client_id to check. (Optional)
        start_date: The start date in format "YYYY-MM-DD".
        end_date: The end date in format "YYYY-MM-DD".

    Returns:
        dict: A dictionary with the client_id, start_date, end_date, and create_report keys. Indicating the
        report information and wether it was generated or not.
    """

    try:
        start_date_dt = pd.to_datetime(start_date, format="%Y-%m-%d")
        end_date_dt = pd.to_datetime(end_date, format="%Y-%m-%d")
        df = df[df["client_id"] == client_id]
        df.loc[:, "date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_date_dt) & (df["date"] <= end_date_dt)]
        df = df.sort_values("date")
        df = df.reset_index(drop=True)

        _ = earnings_and_expenses(df, client_id, start_date, end_date, plot=True)
        _ = expenses_summary(df, client_id, start_date, end_date, plot=True)
        cash_flow_summary_df = cash_flow_summary(df, client_id, start_date, end_date)

        # check if cash_flow_summary_df is empty
        if cash_flow_summary_df.empty:
            raise ValueError("Cash flow summary is empty")

        construct_report(cash_flow_summary_df)

        report_created = True
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        report_created = False
    finally:
        return {
            "start_date": start_date,
            "end_date": end_date,
            "client_id": client_id,
            "create_report": report_created
            }


def construct_report(cash_flow_summary_df):
    """
    This function should fetch the reporst for earnings_and_expenses and expenses_summary under the reports/figures folder.
    This should be loaded into a PDF report together with the cash_flow_summary_df.
    """
    # get figures
    earnings_fig = "reports/figures/earnings_and_expenses.png"
    expenses_fig = "reports/figures/expenses_summary.png"

    # create a new PDF
    doc = SimpleDocTemplate("reports/report.pdf")
    styles = getSampleStyleSheet()
    elements = []

    # add title
    elements.append(Paragraph("Financial Report", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))

    # add earnings and expenses figure
    elements.append(Paragraph("Earnings and Expenses", styles['Heading2']))
    elements.append(Image(earnings_fig, width=6 * inch, height=3 * inch))
    elements.append(Spacer(1, 0.2 * inch))

    # add expenses summary figure
    elements.append(Paragraph("Expenses Summary", styles['Heading2']))
    elements.append(Image(expenses_fig, width=6 * inch, height=3 * inch))
    elements.append(Spacer(1, 0.2 * inch))

    # add cash flow summary
    elements.append(Paragraph("Cash Flow Summary", styles['Heading2']))
    data = [cash_flow_summary_df.columns.tolist()] + cash_flow_summary_df.values.tolist()
    table = Table(data)
    elements.append(table)

    # build the PDF
    doc.build(elements)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Testing tools")
    df = get_transactions_dataset()
    client_id = 122
    start_date = "2017-04-01"
    end_date = "2017-04-30"
    generate_report_output = generate_report.run({'df': df, 'client_id': client_id, 'start_date': start_date, 'end_date': end_date})
    logging.info(f"Report generated: {generate_report_output}")
    # check if file exists
    import os
    assert os.path.exists("reports/report.pdf"), "Report is not present"


    logging.info("Tools tested")
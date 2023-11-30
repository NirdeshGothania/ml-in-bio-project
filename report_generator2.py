from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle


def generate_report(k, accuracy_scores, scores_precision, scores_recall, scores_f1, overall_metrics, best_model, feature_selection_method, normalization_method, cross_validation_method):


    pdf_filename = 'report.pdf'
    pdf = canvas.Canvas(pdf_filename, pagesize=letter)
    
    # Add a title to the report
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(300, 750, "Machine Learning Model Report")

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawCentredString(300, 730, "Nirdesh Gothania (CS21B1016)")

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 700, "Model Details")


    model_details0 = f"Model: {str(best_model)}"
    model_details1 = f"Standardization/Normalization Technique: {str(normalization_method)}"
    model_details2 = f"Feature Selection Technique: {str(feature_selection_method)}"
    model_details3 = f"Cross Validation Technique: {str(cross_validation_method)}"

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 650, model_details0)
    pdf.drawString(50, 600, model_details1)
    pdf.drawString(50, 550, model_details2)
    pdf.drawString(50, 500, model_details3)

    pdf.showPage()


    plt.plot(k,accuracy_scores)
    plt.title('Accuracy Scores across Cross-Validation Folds')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.savefig("accuracy_plot.png", format='png')
    plt.close()


    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 700, "Accuracy Scores across Cross-Validation Folds")

    pdf.drawInlineImage("accuracy_plot.png", 50, 500, width=400, height=200)


    pdf.showPage()

    plt.plot(k,scores_precision)
    plt.title('Precision Scores across Cross-Validation Folds')
    plt.xlabel('Fold')
    plt.ylabel('Precision')
    plt.savefig("precision_plot.png", format='png')
    plt.close()

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 700, "Precision Scores across Cross-Validation Folds")


    pdf.drawInlineImage("precision_plot.png", 50, 500, width=400, height=200)


    pdf.showPage()

    plt.plot(k,scores_recall)
    plt.title('Recall Scores across Cross-Validation Folds')
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.savefig("recall_plot.png", format='png')
    plt.close()


    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 700, "Recall Scores across Cross-Validation Folds")

    pdf.drawInlineImage("recall_plot.png", 50, 500, width=400, height=200)

    pdf.showPage()

    plt.plot(k,scores_f1)
    plt.title('F1 Scores across Cross-Validation Folds')
    plt.xlabel('Fold')
    plt.ylabel('F1 Scores')
    plt.savefig("f1_plot.png", format='png')
    plt.close()


    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 700, "F1 Scores across Cross-Validation Folds")

    pdf.drawInlineImage("f1_plot.png", 50, 500, width=400, height=200)


    pdf.showPage()

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 700, "Overall Metrics for the Blind Dataset")

    metrics_data = [['Metric', 'Score'],
                ['Accuracy', overall_metrics['accuracy']],
                ['Precision', overall_metrics['precision']],
                ['Recall', overall_metrics['recall']],
                ['F1 Score', overall_metrics['f1_score']]]
    table_data = metrics_data[1:]


    style = TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.black)])


    table = Table(table_data, colWidths=80, rowHeights=20, style=style)


    table.wrapOn(pdf, 0, 0)
    table.drawOn(pdf, 50, 600)

    pdf.save()
    print(f"Report generated successfully: {pdf_filename}")

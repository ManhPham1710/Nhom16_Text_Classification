@echo off
chcp 65001 >nul
echo ============================================
echo NHOM 16 - VIETNAMESE TEXT CLASSIFICATION
echo ============================================
if not exist ".venv\Scripts\python.exe" (
    echo Chua co moi truong ao. Dang tao .venv...
    py -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Cai dat hoan tat.
echo Chay huan luyen: python train.py
echo Danh gia: python evaluate.py
echo Giao dien: streamlit run app.py
pause

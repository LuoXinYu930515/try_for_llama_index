rmdir /s /q "chroma_db"
del output.txt
del newstas
python -u chatbot.py > output.txt 2>&1

@echo off
if not exist wikitext_data mkdir wikitext_data
cd wikitext_data
echo Downloading WikiText-2 (Insecure Mode)...
curl -k -L -o wikitext.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
if %errorlevel% neq 0 (
    echo Download failed!
    exit /b 1
)
echo Payload Received.
cd ..

# Troubleshooting HTTP 400 Error on CSV Upload

## What is HTTP 400 Error?

**HTTP 400 (Bad Request)** means the server rejected your file because there's something wrong with the file content, format, or encoding - **not** because of file size.

---

## ✅ Required CSV Format

Your CSV **must** have these columns:

### **Minimum Required:**
- `customer_no` - Customer identification number

### **Recommended Columns:**
- `CustomerName` - Customer name
- `beneficiary_name` - Beneficiary name
- `amount` - Transaction amount (numeric)
- `transfer_type` - Type of transfer (e.g., INTERNATIONAL_PAYMENT, TOP-UP)
- `createdDateTime` - Transaction timestamp (format: `YYYY-MM-DD HH:MM:SS`)
- `reference_no` - Transaction reference number

---

## 🔍 Common Causes & Solutions

### 1. **File Encoding Issues** (Most Common)

**Problem:** CSV contains special characters or is not UTF-8 encoded.

**Solution:**
```bash
# On Mac/Linux - Convert to UTF-8
iconv -f ISO-8859-1 -t UTF-8 input.csv > output_utf8.csv

# On Windows - Use PowerShell
Get-Content "input.csv" -Encoding UTF8 | Set-Content "output_utf8.csv"
```

**Or in Excel:**
1. Open the CSV file
2. File → Save As
3. Choose "CSV UTF-8 (Comma delimited) (*.csv)"
4. Save and try uploading again

---

### 2. **Filename Issues**

**Problem:** Spaces or special characters in filename cause issues.

**Your file:** `All Transfers as of November 04_CSV.csv`

**Solution:** Rename to remove spaces:
```bash
# Rename file
mv "All Transfers as of November 04_CSV.csv" "All_Transfers_Nov04.csv"
```

**Best practices for filenames:**
- ✅ Use underscores: `transaction_data_2024.csv`
- ✅ Use hyphens: `transaction-data-2024.csv`
- ❌ Avoid spaces: `transaction data 2024.csv`
- ❌ Avoid special chars: `transaction@data#2024.csv`

---

### 3. **CSV Format Issues**

**Problem:** Wrong delimiter, malformed CSV, or extra commas.

**Check your CSV:**
```bash
# View first 5 lines on Mac/Linux
head -5 your_file.csv

# View first 5 lines on Windows
Get-Content your_file.csv -Head 5
```

**Common issues:**
- Using semicolon (`;`) instead of comma (`,`)
- Extra commas in data fields (not properly quoted)
- Mixed line endings (Windows `\r\n` vs Unix `\n`)

**Solution - Clean CSV in Python:**
```python
import pandas as pd

# Try reading with different parameters
df = pd.read_csv('your_file.csv', 
                 encoding='utf-8',
                 sep=',',  # or ';' if semicolon-delimited
                 quotechar='"',
                 on_bad_lines='skip')  # Skip problematic lines

# Save cleaned version
df.to_csv('cleaned_file.csv', index=False, encoding='utf-8')
```

---

### 4. **Date Format Issues**

**Problem:** `createdDateTime` column has invalid dates.

**Expected format:** `YYYY-MM-DD HH:MM:SS`

**Examples:**
- ✅ `2024-11-04 14:30:00`
- ✅ `2024-01-15 09:15:30`
- ❌ `11/04/2024 2:30 PM` (US format)
- ❌ `04-Nov-2024 14:30` (Text month)

**Solution - Fix dates in Excel:**
1. Select the `createdDateTime` column
2. Format → Custom
3. Use format: `yyyy-mm-dd hh:mm:ss`
4. Save as CSV UTF-8

**Solution - Fix in Python:**
```python
import pandas as pd

df = pd.read_csv('your_file.csv')

# Convert various date formats to standard format
df['createdDateTime'] = pd.to_datetime(df['createdDateTime'], 
                                        errors='coerce',
                                        infer_datetime_format=True)

# Format as string in correct format
df['createdDateTime'] = df['createdDateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Remove rows with invalid dates (NaT)
df = df.dropna(subset=['createdDateTime'])

df.to_csv('fixed_dates.csv', index=False, encoding='utf-8')
```

---

### 5. **Missing or Extra Columns**

**Problem:** Column names don't match expectations.

**Check column names:**
```python
import pandas as pd
df = pd.read_csv('your_file.csv', nrows=1)
print(df.columns.tolist())
```

**Required minimum:**
- Must have: `customer_no`

**Common mistakes:**
- `customer_no` vs `customerno` vs `customer_id`
- Extra spaces: `customer_no ` (space at end)
- Case sensitive on some systems

---

### 6. **File Corruption**

**Problem:** File got corrupted during download/transfer.

**Test:**
```bash
# Check if file opens properly
cat your_file.csv | head -10

# Check file size
ls -lh your_file.csv

# Check for null bytes (corruption indicator)
grep -a '\x00' your_file.csv
```

**Solution:**
- Re-download or re-export the CSV
- Try opening in text editor (not Excel) to verify content

---

## 🚀 Quick Validation Script

Create this Python script to validate your CSV before uploading:

```python
import pandas as pd
import sys

def validate_csv(filename):
    """Validate CSV format for AML Dashboard"""
    print(f"🔍 Validating {filename}...")
    
    try:
        # Try reading CSV
        df = pd.read_csv(filename, encoding='utf-8', nrows=5)
        print(f"✅ File readable")
        print(f"📊 Columns found: {df.columns.tolist()}")
        
        # Check required columns
        if 'customer_no' not in df.columns:
            print("❌ ERROR: Missing required column 'customer_no'")
            return False
        print("✅ Required column 'customer_no' found")
        
        # Check date column if exists
        if 'createdDateTime' in df.columns:
            try:
                pd.to_datetime(df['createdDateTime'])
                print("✅ Date format valid")
            except:
                print("⚠️  WARNING: Some dates may be invalid")
        
        # Check file size
        import os
        size_mb = os.path.getsize(filename) / (1024*1024)
        print(f"📦 File size: {size_mb:.2f} MB")
        
        if size_mb > 1000:
            print("❌ ERROR: File too large (>1GB)")
            return False
        
        print("\n✅ CSV validation passed!")
        return True
        
    except UnicodeDecodeError:
        print("❌ ERROR: File encoding issue. Try converting to UTF-8")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_csv.py your_file.csv")
    else:
        validate_csv(sys.argv[1])
```

**Usage:**
```bash
python validate_csv.py "All Transfers as of November 04_CSV.csv"
```

---

## 🔧 Server-Side Fixes (AWS EC2)

### Update Streamlit Config

Add to `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 1000
maxMessageSize = 1000
enableXsrfProtection = false
enableCORS = false
fileWatcherType = "none"

[browser]
gatherUsageStats = false

[runner]
fastReruns = true
magicEnabled = false
```

### Update NGINX Config (if applicable)

Add to `/etc/nginx/nginx.conf`:

```nginx
http {
    client_max_body_size 1000M;
    client_body_timeout 300s;
    client_body_buffer_size 128k;
    
    # Handle large headers
    large_client_header_buffers 4 32k;
}
```

---

## 📝 Sample Valid CSV

Create this test file to verify the system works:

```csv
customer_no,CustomerName,beneficiary_name,amount,transfer_type,createdDateTime,reference_no
CUST001,John Doe,Beneficiary A,5000.00,INTERNATIONAL_PAYMENT,2024-11-04 10:30:00,REF001
CUST002,Jane Smith,,1000.00,TOP-UP,2024-11-04 11:15:00,REF002
CUST003,Bob Johnson,Beneficiary B,7500.00,WALLET_TO_WALLET_EXTERNAL,2024-11-04 12:00:00,REF003
```

Save as `test_upload.csv` and try uploading.

---

## 🐛 Debugging Steps

### 1. Check Browser Console
1. Open browser Developer Tools (F12)
2. Go to Console tab
3. Try uploading file
4. Look for detailed error messages

### 2. Check Network Tab
1. Open Developer Tools (F12)
2. Go to Network tab
3. Try uploading file
4. Click on the failed request
5. Check "Response" tab for detailed error

### 3. Check Server Logs (EC2)

```bash
# If using Docker
docker-compose logs -f aml-dashboard

# If using systemd
sudo journalctl -u streamlit -f

# NGINX error logs
sudo tail -f /var/log/nginx/error.log
```

---

## ✅ Checklist Before Upload

- [ ] File is UTF-8 encoded
- [ ] Filename has no spaces or special characters  
- [ ] CSV has `customer_no` column
- [ ] Dates are in format: `YYYY-MM-DD HH:MM:SS`
- [ ] File size < 1GB
- [ ] No null bytes or corruption
- [ ] Opens correctly in text editor
- [ ] Commas are delimiters (not semicolons)
- [ ] No extra trailing commas

---

## 🆘 Still Not Working?

If you've tried everything above, please provide:

1. **First 5 lines of your CSV:**
   ```bash
   head -5 your_file.csv
   ```

2. **Column names:**
   ```python
   import pandas as pd
   print(pd.read_csv('your_file.csv', nrows=1).columns.tolist())
   ```

3. **File info:**
   ```bash
   file your_file.csv
   ls -lh your_file.csv
   ```

4. **Browser console error** (F12 → Console)

5. **Server logs** (from EC2)


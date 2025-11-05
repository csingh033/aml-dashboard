#!/usr/bin/env python3
"""
CSV Validator for AML Dashboard
Validates CSV files before uploading to ensure they meet requirements
"""

import pandas as pd
import sys
import os


def validate_csv(filename):
    """Validate CSV format for AML Dashboard"""
    print(f"\n{'='*60}")
    print(f"🔍 Validating CSV: {filename}")
    print(f"{'='*60}\n")

    errors = []
    warnings = []

    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"❌ ERROR: File '{filename}' not found")
            return False

        # Check file size
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"📦 File size: {size_mb:.2f} MB")

        if size_mb > 1000:
            errors.append("File too large (>1GB)")
        elif size_mb > 500:
            warnings.append(f"File is quite large ({size_mb:.2f} MB)")

        # Try reading CSV with different encodings
        encodings_to_try = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
        df = None
        used_encoding = None

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(filename, encoding=encoding, nrows=100)
                used_encoding = encoding
                print(f"✅ File readable (encoding: {encoding})")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"⚠️  Error with {encoding}: {str(e)}")

        if df is None:
            errors.append("Cannot read file with any standard encoding")
            print(f"\n{'='*60}")
            print("❌ VALIDATION FAILED")
            print(f"{'='*60}\n")
            for error in errors:
                print(f"  ❌ {error}")
            return False

        if used_encoding != "utf-8":
            warnings.append(
                f"File uses {used_encoding} encoding. Recommend converting to UTF-8"
            )

        # Check columns
        print(f"\n📊 Columns found ({len(df.columns)}):")
        for col in df.columns:
            print(f"   - {col}")

        # Check required columns
        required_columns = ["customer_no"]
        missing_required = [col for col in required_columns if col not in df.columns]

        if missing_required:
            errors.append(f"Missing required columns: {', '.join(missing_required)}")
        else:
            print(f"\n✅ Required column 'customer_no' found")

        # Check recommended columns
        recommended_columns = [
            "CustomerName",
            "beneficiary_name",
            "amount",
            "transfer_type",
            "createdDateTime",
            "reference_no",
        ]
        missing_recommended = [
            col for col in recommended_columns if col not in df.columns
        ]

        if missing_recommended:
            print(f"\n⚠️  Missing recommended columns:")
            for col in missing_recommended:
                print(f"   - {col}")
                warnings.append(f"Missing recommended column: {col}")
        else:
            print(f"\n✅ All recommended columns present")

        # Validate date column if exists
        if "createdDateTime" in df.columns:
            print(f"\n🕒 Validating date format...")
            try:
                dates = pd.to_datetime(df["createdDateTime"], errors="coerce")
                null_dates = dates.isna().sum()

                if null_dates == len(df):
                    errors.append("All dates are invalid in 'createdDateTime'")
                elif null_dates > 0:
                    warnings.append(
                        f"{null_dates} out of {len(df)} sample rows have invalid dates"
                    )
                    print(f"⚠️  {null_dates}/{len(df)} sample rows have invalid dates")
                else:
                    print(f"✅ Date format valid")
                    print(
                        f"   Date range: {dates.min()} to {dates.max()}"
                    )
            except Exception as e:
                warnings.append(f"Date validation issue: {str(e)}")

        # Validate amount column if exists
        if "amount" in df.columns:
            print(f"\n💰 Validating amounts...")
            try:
                amounts = pd.to_numeric(df["amount"], errors="coerce")
                invalid_amounts = amounts.isna().sum()

                if invalid_amounts > 0:
                    warnings.append(
                        f"{invalid_amounts} out of {len(df)} sample rows have invalid amounts"
                    )
                else:
                    print(
                        f"✅ Amount format valid (range: {amounts.min():.2f} to {amounts.max():.2f})"
                    )
            except Exception as e:
                warnings.append(f"Amount validation issue: {str(e)}")

        # Check for empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            warnings.append(f"{empty_rows} completely empty rows found")

        # Check for duplicate reference numbers if column exists
        if "reference_no" in df.columns:
            duplicates = df["reference_no"].duplicated().sum()
            if duplicates > 0:
                warnings.append(f"{duplicates} duplicate reference numbers found")

        # Read full file to get total row count
        try:
            full_df = pd.read_csv(filename, encoding=used_encoding, usecols=[0])
            total_rows = len(full_df)
            print(f"\n📈 Total rows in file: {total_rows:,}")
        except Exception:
            print(f"\n📈 Sample rows validated: {len(df)}")

        # Print summary
        print(f"\n{'='*60}")
        if errors:
            print("❌ VALIDATION FAILED")
            print(f"{'='*60}\n")
            print("Errors:")
            for error in errors:
                print(f"  ❌ {error}")
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  ⚠️  {warning}")
            return False
        else:
            if warnings:
                print("⚠️  VALIDATION PASSED WITH WARNINGS")
                print(f"{'='*60}\n")
                print("Warnings:")
                for warning in warnings:
                    print(f"  ⚠️  {warning}")
                print(
                    "\n💡 The file will likely work, but you may want to address these warnings."
                )
            else:
                print("✅ VALIDATION PASSED")
                print(f"{'='*60}\n")
                print("🎉 Your CSV file looks good and ready to upload!")
            return True

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("\n" + "=" * 60)
        print("CSV Validator for AML Dashboard")
        print("=" * 60)
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <csv_file>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} transactions.csv")
        print()
        sys.exit(1)

    filename = sys.argv[1]
    success = validate_csv(filename)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


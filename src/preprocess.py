import pandas as pd

def load_dataset(path: str, dataset_type: str):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Standardize date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Clean location columns
    for col in ['state', 'district']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Clean pincode
    if 'pincode' in df.columns:
        df['pincode'] = df['pincode'].astype(str).str.zfill(6)

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Add derived columns based on dataset type
    if dataset_type == 'enrolment':
        enrol_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
        if all(col in df.columns for col in enrol_cols):
            df['total_enrolment'] = df[enrol_cols].sum(axis=1)
        else:
            print(f"Missing enrolment columns: {enrol_cols}")
    elif dataset_type == 'biometric':
        bio_cols = [col for col in df.columns if 'bio' in col]
        if bio_cols:
            df['total_biometric'] = df[bio_cols].sum(axis=1)
        else:
            print("No biometric columns found.")
    elif dataset_type == 'demographic':
        demo_cols = [col for col in df.columns if 'demo' in col]
        if demo_cols:
            df['total_demographic'] = df[demo_cols].sum(axis=1)
        else:
            print("No demographic columns found.")

    return df

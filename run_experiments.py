import os, glob
from sklearn.model_selection import train_test_split

from beatbotLite import run_training, build_parser

def build_cohort():
    pattern = os.path.join("VT-data", "Patient_*.h5")
    allfiles = glob.glob(pattern)
    files = [f for f in allfiles if not os.path.basename(f).endswith('_o.h5')]
    numbers = [os.path.basename(f) for f in files]
    cohort = sorted(int(fname.split("_")[1].split(".")[0]) for fname in numbers)

    evaluation_idx = [cohort[-1]]
    remaining = [pid for pid in cohort if pid not in evaluation_idx]
    train_idx, validation_idx = train_test_split(remaining, test_size=0.2, random_state=42)

    train_patients = ",".join(str(i) for i in sorted(train_idx))
    eval_patients = ",".join(str(i) for i in sorted(validation_idx + evaluation_idx))
    return train_patients, eval_patients

def main():
    parser = build_parser()
    args = parser.parse_args([])
    train_patients, eval_patients = build_cohort()

    args.train_patients = train_patients
    args.eval_patients = eval_patients

    run_training(args)

if __name__ == "__main__":
    main()
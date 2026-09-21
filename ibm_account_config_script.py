from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("IBM_QUANTUM_API_KEY")
crn = os.getenv("IBM_QUANTUM_CRN")

if not api_key:
    raise ValueError("IBM_QUANTUM_API_KEY not found in .env")

if not crn:
    raise ValueError("IBM_QUANTUM_CRN not found in .env")

# Save credentials locally
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=api_key,
    instance=crn,
    overwrite=True,
)

print("Account saved successfully.\n")

# Verify by loading the saved account
service = QiskitRuntimeService()

print("Available instances:")
print(service.instances())

print("\nAvailable backends:")
for backend in service.backends():
    print(f"- {backend.name}")
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SchemeDocument:
    """Represents a government scheme document"""
    id: str
    title: str
    description: str
    category: str
    eligibility: List[str]
    benefits: List[str]
    application_process: str
    required_documents: List[str]
    contact_info: str
    website: str
    state: Optional[str] = None
    
    def to_text(self) -> str:
        """Convert scheme to searchable text"""
        return f"""
Scheme: {self.title}
Category: {self.category}
State: {self.state or 'All India'}

Description: {self.description}

Eligibility Criteria:
{chr(10).join(f"- {item}" for item in self.eligibility)}

Benefits:
{chr(10).join(f"- {item}" for item in self.benefits)}

Application Process: {self.application_process}

Required Documents:
{chr(10).join(f"- {item}" for item in self.required_documents)}

Contact: {self.contact_info}
Website: {self.website}
"""



def initialize_mock_schemes():
    """Initialize with mock Indian agricultural schemes"""
    mock_schemes = [
        SchemeDocument(
            id="PM-KISAN-001",
            title="Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
            description="Direct income support scheme providing ₹6000 per year to all landholding farmer families in three equal installments of ₹2000 each.",
            category="Income Support",
            eligibility=[
                "All landholding farmer families",
                "Small and marginal farmers with cultivable land",
                "Family defined as husband, wife and minor children"
            ],
            benefits=[
                "₹6000 per year in three installments",
                "Direct bank transfer (DBT)",
                "No limit on family income"
            ],
            application_process="Register online at pmkisan.gov.in or visit nearest Common Service Center (CSC) with land records and Aadhaar card.",
            required_documents=[
                "Aadhaar card",
                "Bank account details",
                "Land ownership documents"
            ],
            contact_info="PM-KISAN Helpline: 011-24300606, Email: pmkisan-ict@gov.in",
            website="https://pmkisan.gov.in",
            state=None
        ),
        SchemeDocument(
            id="PMFBY-001",
            title="Pradhan Mantri Fasal Bima Yojana (PMFBY)",
            description="Crop insurance scheme providing financial support to farmers in case of crop failure due to natural calamities, pests, or diseases.",
            category="Crop Insurance",
            eligibility=[
                "All farmers including sharecroppers and tenant farmers",
                "Compulsory for farmers availing crop loan",
                "Voluntary for non-loanee farmers"
            ],
            benefits=[
                "Maximum premium: 2% for Kharif, 1.5% for Rabi, 5% for commercial crops",
                "Coverage from sowing to post-harvest",
                "Claims settled within 2 months"
            ],
            application_process="Apply through banks, insurance companies, or online at pmfby.gov.in before crop sowing cutoff date.",
            required_documents=[
                "Land records or tenancy agreement",
                "Aadhaar card",
                "Bank account details",
                "Sowing certificate"
            ],
            contact_info="PMFBY Helpline: 011-23382012, Toll-free: 18001801551",
            website="https://pmfby.gov.in",
            state=None
        ),
        SchemeDocument(
            id="MICRO-IRR-001",
            title="Per Drop More Crop - Micro Irrigation Scheme",
            description="Subsidy for adoption of micro irrigation systems (drip and sprinkler) to improve water use efficiency and increase crop productivity.",
            category="Irrigation",
            eligibility=[
                "All categories of farmers",
                "Farmers with assured water source",
                "Priority to small and marginal farmers"
            ],
            benefits=[
                "Subsidy: 55% for small/marginal farmers, 45% for other farmers",
                "Additional 10% for SC/ST farmers",
                "Up to ₹50,000 subsidy per hectare"
            ],
            application_process="Apply online through state agriculture department portal or visit District Agriculture Office with land records and water source proof.",
            required_documents=[
                "Land ownership documents",
                "Water source availability certificate",
                "Aadhaar card",
                "Bank account details",
                "Cost estimate from approved vendor"
            ],
            contact_info="Contact District Agriculture Officer or State Agriculture Department",
            website="https://pmksy.gov.in",
            state=None
        ),
        SchemeDocument(
            id="KCC-001",
            title="Kisan Credit Card (KCC)",
            description="Credit facility to farmers for agricultural needs including crop production, asset maintenance, consumption needs, and working capital.",
            category="Credit/Loan",
            eligibility=[
                "Farmers (individual/joint borrowers)",
                "Tenant farmers and sharecroppers",
                "Self Help Groups or Joint Liability Groups of farmers"
            ],
            benefits=[
                "Flexible credit limit based on cropping pattern",
                "Interest subvention: 3% per annum",
                "Additional 3% incentive for timely repayment",
                "Insurance cover up to ₹50,000"
            ],
            application_process="Visit nearest bank branch (commercial, RRB, or cooperative) with land documents and KYC documents.",
            required_documents=[
                "Aadhaar card",
                "PAN card",
                "Land records",
                "Passport size photos"
            ],
            contact_info="Contact nearest bank branch or call 1800-180-1551",
            website="https://www.nabard.org/kcc.aspx",
            state=None
        ),
        SchemeDocument(
            id="PKVY-001",
            title="Paramparagat Krishi Vikas Yojana (PKVY)",
            description="Scheme to promote organic farming and certify organic produce through cluster approach and PGS certification.",
            category="Organic Farming",
            eligibility=[
                "Groups of farmers forming clusters (minimum 50 acres)",
                "Farmers willing to convert to organic farming",
                "Participatory Guarantee System (PGS) certification required"
            ],
            benefits=[
                "₹50,000 per hectare over 3 years",
                "Support for organic inputs, seeds, bio-fertilizers",
                "Market linkage support",
                "Training and capacity building"
            ],
            application_process="Apply through District Agriculture Office or form cluster of minimum 20 farmers with 50 acres total land.",
            required_documents=[
                "Land records",
                "Aadhaar card",
                "Cluster formation documents",
                "Bank account details (individual and group)"
            ],
            contact_info="Contact District Agriculture Officer or State Organic Farming Mission",
            website="https://pgsindia-ncof.gov.in",
            state=None
        ),
        SchemeDocument(
            id="SMAM-001",
            title="Sub-Mission on Agricultural Mechanization (SMAM)",
            description="Financial assistance for purchase of agricultural machinery and equipment to increase farm mechanization and productivity.",
            category="Farm Mechanization",
            eligibility=[
                "Individual farmers",
                "Custom Hiring Centers",
                "Farmer Producer Organizations (FPOs)",
                "Priority to SC/ST, small and marginal farmers"
            ],
            benefits=[
                "40-50% subsidy on farm equipment",
                "Additional 10% for SC/ST farmers",
                "Support for establishment of Custom Hiring Centers",
                "Subsidy on tractors, harvesters, tillers, etc."
            ],
            application_process="Apply online through Direct Benefit Transfer Agriculture portal (DBT Agriculture) with quotations from authorized dealers.",
            required_documents=[
                "Aadhaar card",
                "Land ownership documents",
                "Bank account details",
                "Caste certificate (if applicable)",
                "Equipment quotation from dealer"
            ],
            contact_info="Contact District Agriculture Office or visit dbtAgriculture portal",
            website="https://agrimachinery.nic.in",
            state=None
        ),
        SchemeDocument(
            id="NFSM-001",
            title="National Food Security Mission (NFSM)",
            description="Support for increasing production of rice, wheat, pulses, coarse cereals and commercial crops through distribution of quality seeds, farm equipment, and technical assistance.",
            category="Production Enhancement",
            eligibility=[
                "All farmers in identified districts",
                "Focus on rainfed and low productivity areas",
                "Farmers growing targeted crops"
            ],
            benefits=[
                "50% subsidy on quality seeds",
                "Support for soil health management",
                "Assistance for farm equipment",
                "Training and demonstrations",
                "Support for water harvesting structures"
            ],
            application_process="Register through District Agriculture Office or Krishi Vigyan Kendra (KVK) during crop season.",
            required_documents=[
                "Aadhaar card",
                "Land records",
                "Bank account details"
            ],
            contact_info="Contact District Agriculture Officer or nearest KVK",
            website="https://nfsm.gov.in",
            state=None
        ),
        SchemeDocument(
            id="RKVY-001",
            title="Rashtriya Krishi Vikas Yojana (RKVY)",
            description="State-level agricultural development scheme providing flexibility to states to plan and execute agriculture projects based on local needs.",
            category="General Development",
            eligibility=[
                "State governments submit proposals",
                "Individual farmers benefit through state schemes",
                "Farmer groups and FPOs"
            ],
            benefits=[
                "Funding for infrastructure development",
                "Support for value addition and marketing",
                "Assistance for farmer producer organizations",
                "Innovation and technology adoption support"
            ],
            application_process="Check with State Agriculture Department for state-specific sub-schemes under RKVY framework.",
            required_documents=[
                "Varies by state-specific sub-scheme"
            ],
            contact_info="Contact State Agriculture Department",
            website="https://rkvy.nic.in",
            state=None
        ),
        SchemeDocument(
            id="SOIL-HEALTH-001",
            title="Soil Health Card Scheme",
            description="Provides soil health cards to farmers with crop-wise recommendations on nutrients and fertilizers required for their farm.",
            category="Soil Health",
            eligibility=[
                "All farmers",
                "Cards issued every 2 years"
            ],
            benefits=[
                "Free soil testing",
                "Customized fertilizer recommendations",
                "Reduction in fertilizer cost",
                "Improved soil health and productivity"
            ],
            application_process="Visit nearest Soil Testing Laboratory or Agriculture Office with soil samples. Online tracking available.",
            required_documents=[
                "Aadhaar card",
                "Land records",
                "Soil sample (collected as per guidelines)"
            ],
            contact_info="Contact District Soil Testing Laboratory",
            website="https://soilhealth.dac.gov.in",
            state=None
        ),
        SchemeDocument(
            id="KAR-RAITHA-001",
            title="Raitha Sanjeevini (Karnataka)",
            description="Karnataka state scheme providing comprehensive crop insurance coverage with minimal premium for all farmers.",
            category="Crop Insurance",
            eligibility=[
                "All farmers in Karnataka",
                "Covers all crops in all seasons"
            ],
            benefits=[
                "Comprehensive insurance coverage",
                "Nominal premium: ₹1 per season",
                "Coverage up to ₹2 lakhs",
                "Quick claim settlement"
            ],
            application_process="Automatically enrolled through land records. Visit nearest Raitha Samparka Kendra for updates.",
            required_documents=[
                "Land records",
                "Aadhaar card"
            ],
            contact_info="Karnataka Agriculture Department: 080-22212825",
            website="https://raitamitra.karnataka.gov.in",
            state="Karnataka"
        )
    ]
    
    return mock_schemes
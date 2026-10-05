"""
schema.py — the DD intake form, defined once.

Every downstream shape is generated from FIELDS by schema_export.py: the
frontend form definition, the Google Doc layout, the HubSpot property map, the
prospect-row digest and the profile source text. Editing a question means
editing one Field here and nothing else.

FIELDS is transcribed from the live HubSpot "New Due Diligence Form"
(f6099a96-e44d-45ec-b021-aaa612732100) via scripts/intake/pull_hubspot_form.py:
labels and options verbatim, required flags as the form has them, and each
answer written to the SAME HubSpot property the form writes, so staff views,
lists and history stay continuous. Deliberate departures from the form:
  * conditions — the form has none; the intake hides the health questions
    unless a health vertical is ticked, the investor questions unless funding
    includes Outside Investors, and the "if yes" follow-ups unless answered
    yes. A hidden question is neither required nor stored.
  * 'Option 5' (an unfinished placeholder) is dropped from "Who pays".
  * Sections group the form's single page for the wizard (DD_INTAKE_PLAN §3).

Ids in this file are load-bearing — profile_writer, hubspot_sync and
drive_folder read the core ones by name — and never reused.
"""

from dataclasses import asdict, dataclass, field

# ── Sections (display order) ────────────────────────────────────────────────

SECTIONS: list[tuple[str, str]] = [
    ('1_company',            'Company & eligibility'),
    ('2_technology',         'Technology'),
    ('2a_nih',               'Health & life sciences'),
    ('3_evidence',           'Evidence'),
    ('4_team',               'Team'),
    ('5_funding',            'Funding goals'),
    ('5_company_funding',    'Company funding & milestones'),
    ('5a_federal',           'Federal traction'),
    ('6_commercialization',  'Commercialization'),
    ('7_uploads',            'Documents'),
]
SECTION_IDS    = [s for s, _ in SECTIONS]
SECTION_TITLES = dict(SECTIONS)

TYPES = ('text', 'textarea', 'single_select', 'multi_select', 'email', 'phone',
         'url', 'file')

# profile_role — what a founder answer is allowed to become in the Matcher.
#   capability: feeds the profile's source text (aspect_profile's `intake` source)
#   intention:  forward-looking (goals, plans, adjacent markets) — goes to
#               extracted['notable_updates'], read only by the unexplored-market
#               pass, never as capability (same rule as Fathom/Drive material)
#   exclude:    eligibility, finances and personal data — kept in intake_data
#               and the Google Doc only, never sent to a model
PROFILE_ROLES = ('capability', 'intention', 'exclude')


@dataclass(frozen=True)
class Field:
    id: str
    section: str
    label: str
    type: str
    required: bool = False
    help_text: str = ''
    options: tuple[str, ...] = ()
    # HubSpot's internal enumeration values, parallel to `options`, where they
    # differ from the labels (most dropdowns on this form store opaque codes).
    option_values: tuple[str, ...] = ()
    # Visibility: clauses OR-ed together, each {'field': <id>, 'any_of': [values]};
    # None = always shown.
    condition: tuple[dict, ...] | None = None
    ai_fillable: bool = False        # Phase 2 — confirmed by the extraction eval
    enrichable: bool = False         # Phase 3 — federal lookups
    profile_role: str = 'exclude'
    hubspot_property: str | None = None
    hubspot_object: str = 'company'  # 'company' | 'contact'
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['options'] = list(self.options)
        d['condition'] = list(self.condition) if self.condition else None
        return d

    def hubspot_value(self, label: str) -> str:
        """The value HubSpot stores for one option label."""
        if self.option_values and label in self.options:
            return self.option_values[self.options.index(label)]
        return label


# ── Core field ids (load-bearing) ───────────────────────────────────────────

COMPANY_NAME     = 'company_legal_name'
WEBSITE          = 'website'
CONTACT_FIRST    = 'contact_first_name'
CONTACT_LAST     = 'contact_last_name'
CONTACT_EMAIL    = 'contact_email'
COMPANY_STATE    = 'company_state'
TECH_DESCRIPTION = 'technology_description'
VERTICALS        = 'verticals'

# ── Conditions ──────────────────────────────────────────────────────────────

HEALTH_VERTICALS = ['Health Tech', 'Medtech', 'Biotech']
HEALTH_CONDITION = ({'field': VERTICALS, 'any_of': HEALTH_VERTICALS},)
INVESTOR_CONDITION = ({'field': 'current_funding', 'any_of': ['Outside Investors']},)
ADJACENT_CONDITION = ({'field': 'open_to_adjacent',
                       'any_of': ['Yes', 'Possibly, depending on the project']},)
FEDERAL_CONDITION = ({'field': 'federal_funding_received',
                      'any_of': ['SBIR/STTR', 'Other federal grant or cooperative agreement',
                                 'Federal contract', 'OTA or prototype agreement',
                                 'CRADA or other federal partnership', 'Other']},)

# ── The form ────────────────────────────────────────────────────────────────

FIELDS: list[Field] = [
    Field('contact_first_name', '1_company', 'First Name', 'text', required=True,
          hubspot_property='firstname', hubspot_object='contact'),
    Field('contact_last_name', '1_company', 'Last Name', 'text', required=True,
          hubspot_property='lastname', hubspot_object='contact'),
    Field('contact_email', '1_company', 'Email', 'email', required=True,
          hubspot_property='email', hubspot_object='contact'),
    Field('company_legal_name', '1_company', 'Company Name (LEGAL)', 'text', required=True,
          hubspot_property='company_name__legal_', hubspot_object='company'),
    Field('entity_type', '1_company', 'Entity Type (LLC, C Corp, etc.)', 'text', required=True,
          hubspot_property='entity_type__llc__c_corp__etc__', hubspot_object='contact'),
    Field('website', '1_company', 'Website URL', 'url', required=True,
          ai_fillable=True,
          hubspot_property='website', hubspot_object='contact'),
    Field('company_state', '1_company', 'State/Region', 'text', required=True,
          hubspot_property='state', hubspot_object='company'),
    Field('employee_count', '1_company', 'Number of Employees', 'single_select', required=True,
          options=('1-9', '10-49', '50-249', '250-500', '500+'),
          option_values=('1-9', '10-49', '50-249', '250+', '500+'),
          hubspot_property='numemployees', hubspot_object='contact'),
    Field('us_owned_51', '1_company', 'Is your company primarily (51% or more) owned by US citizens or permanent residents?', 'multi_select', required=True,
          options=('Yes', 'No'),
          hubspot_property='is_your_company_51__or_more_owned_by_us_citizens_or_permanent_residents_', hubspot_object='contact'),
    Field('vc_pe_owned_50', '1_company', 'Does any venture capital, hedge fund, or private equity firm own more than 50% combined?', 'single_select', required=True,
          options=('Yes', 'No', 'Not Sure'),
          hubspot_property='does_any_venture_capital_hedge_fund_or_private_equity_firm_own_more_than_50_combined', hubspot_object='contact'),
    Field('sam_registered', '1_company', 'Registered in SAM.gov with an active UEI?', 'single_select', required=True,
          options=('Yes', 'No', 'Not Sure'),
          enrichable=True,
          hubspot_property='registered_in_samgov_with_an_active_uei', hubspot_object='contact'),
    Field('prior_federal_awards', '1_company', 'Does the company have prior SBIR/STTR or other federal awards?', 'single_select', required=True,
          options=('Yes', 'No', 'Not Sure'),
          enrichable=True,
          hubspot_property='does_the_company_have_prior_sbirsttr_or_other_federal_awards', hubspot_object='contact'),
    Field('technology_description', '2_technology', 'In 2 to 3 sentences, what does your technology do and for whom?', 'textarea', required=True,
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='in_2_to_3_sentences_what_does_your_technology_do_and_for_whom', hubspot_object='contact'),
    Field('stage', '2_technology', 'Which best describes your stage?', 'single_select', required=True,
          options=('Concept Only', 'Lab Prototype', 'Tested in Relevant Environment', 'Pilot with a Customer/End User', 'Commercial Product'),
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='which_best_describes_your_stage', hubspot_object='contact'),
    Field('government_end_user', '2_technology', 'Does the technology have a plausible defense, space, or other government end user?', 'single_select', required=True,
          options=('Yes', 'No', 'Not Sure'),
          ai_fillable=True,
          profile_role='intention',
          hubspot_property='does_the_technology_have_a_plausible_defense_space_or_other_government_end_user', hubspot_object='contact'),
    Field('verticals', '2_technology', 'Verticals', 'multi_select', required=True,
          options=('Advanced Materials & Manufacturing', 'Aerospace & Spacetech', 'Agtech & Foodtech', 'Artificial Intelligence & Machine Learning', 'Biotech', 'Cleantech & Energy', 'Cybersecurity', 'Defense Tech & Dual-Use Tech', 'eXtended Reality', 'Health Tech', 'Medtech', 'Quantum & Photonics', 'Robotics & Autonomous Systems'),
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='verticals', hubspot_object='company'),
    Field('visible_failure', '2_technology', 'What is the visible failure or cost today?', 'textarea',
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='what_is_the_visible_failure_or_cost_today', hubspot_object='contact'),
    Field('urgency', '2_technology', 'What happens over the next 2 to 3 years if this is not solved?', 'textarea', required=True,
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='what_happens_over_the_next_2_to_3_years_if_this_is_not_solved', hubspot_object='contact'),
    Field('competitors', '2_technology', 'What competitors are innovating in this space and why is your technology different/better?', 'text', required=True,
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='what_competitors_are_innovating_in_this_space_and_why_is_your_technology_different_better_', hubspot_object='company'),
    Field('patient_population', '2a_nih', 'Target Patient Population', 'multi_select', required=True,
          options=('Neonatal', 'Infant', 'Pediatric', 'Adult', 'Geriatric'),
          condition=HEALTH_CONDITION,
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='target_patient_population', hubspot_object='company'),
    Field('study_types', '2a_nih', 'What Sort of Studies will be involved?', 'multi_select', required=True,
          options=('Animal Studies', 'Human Subjects', 'Clinical Trial', 'Pre-Clinical Work', 'None of the Above'),
          condition=HEALTH_CONDITION,
          profile_role='intention',
          hubspot_property='what_sort_of_studies_will_be_involved_', hubspot_object='contact'),
    Field('unmet_need', '2a_nih', 'Does the technology address an unmet medical need with no adequate current option, or improve on an existing option?', 'single_select',
          options=('Unmet need', 'Improvement', 'Not Sure'),
          condition=HEALTH_CONDITION,
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='does_the_technology_address_an_unmet_medical_need', hubspot_object='contact'),
    Field('pi_employment', '2a_nih', 'Will the principal investigator be employed more than 50% by the company at the time of award?', 'single_select',
          options=('Yes', 'No', 'Not Sure', 'PI will remain employed at partnering University'),
          condition=HEALTH_CONDITION,
          hubspot_property='will_the_principal_investigator_be_employed_more_than_50__by_the_company_at_the_time_of_award_', hubspot_object='contact'),
    Field('evidence', '3_evidence', 'Check all that exist:', 'multi_select',
          options=('Peer Reviewed Publications', 'Patents Filed or Granted', 'Preliminary or Pilot Data', 'Letters of Intent from Customers', 'Paying Customers', 'Prior Government Contracts or CRADAs', 'None of the Above'),
          option_values=('8syoo8PS0qIsenfpMQYVa', '5s_sLQmUGf1DkcPcrAEBA', 'yWZ4yZOYzplNRgF6KhRaA', 'jK6VHpL9KUu6kD3eFjicJ', 'jA22xRkizVSQWG9rmEuZe', 'fYKSYhPXHf5Nevg4Ye7bI', 'f9JDEbO1G9b17xLLUJ3UK'),
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='check_all_that_exist_', hubspot_object='contact'),
    Field('evidence_links', '3_evidence', 'Optional: Paste links to publications, products, or press', 'textarea',
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='optional__paste_links_to_publications__products__or_press', hubspot_object='contact'),
    Field('technical_lead', '4_team', 'Who would lead the technical work? Name, title, highest degree, and whether they are full-time at the company', 'textarea',
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='who_would_lead_the_technical_work__name__title__highest_degree__and_whether_they_are_full_time_at_t', hubspot_object='contact'),
    Field('in_house_capabilities', '4_team', 'Which of these do you have in-house today?', 'multi_select',
          options=('Technical R&D Lead', 'Regulatory or Quality', 'Biostatistics or Data Analysis', 'Manufacturing or Engineering Scale-up', 'Business Development or Government Sales', 'Grants Administration & Accounting'),
          option_values=('oHqq_gf5pFKJ0ENvi8jPi', 'eW73qTICYS_ZERipUnJUI', '5hYWzQdUMRYwwQY8UVhSJ', 'XvokEtcDwvktQBmHESaGk', 'wTcVwHXPlLV_V5jII31FX', 'qiNGVubaUE-VXNuGqe3GW'),
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='which_of_these_do_you_have_in_house_today_', hubspot_object='contact'),
    Field('open_to_cro', '4_team', 'Are you open to bringing in a CRO, subcontractor, or outside expert to fill gaps?', 'single_select',
          options=('Yes', 'No'),
          option_values=('hutjXUb8K2vGhG8wrhW7W', 'HGkjmeyiECptIJVD2S0uD'),
          hubspot_property='are_you_open_to_bringing_in_a_cro__subcontractor__or_outside_expert_to_fill_gaps_', hubspot_object='contact'),
    Field('research_partner', '4_team', 'Do you have a Partner (e.x., University, National Lab, Research Institution)', 'single_select', required=True,
          options=('Our Partner has already communicated their willingness to work with us on this project.', 'We have a Partner identified but are still discussing the details.', 'We do not have a Partner identified and would like help securing one.', 'We do not have a Partner and are not interested in securing one.'),
          option_values=('7vQJx7z5rDWXajysNsnhG', 'KHtIg0oCN0U1bXjcsd3FW', 'We do not have a Partner identified and would like help securing one.', 'We do not have a Partner and are not interested in securing one.'),
          hubspot_property='do_you_have_a_partner__e_x___university__national_lab__research_institution_', hubspot_object='contact'),
    Field('funding_goals', '5_funding', 'What do you want non-dilutive funding to accomplish over the next 12 to 24 months? (2 to 3 sentences)', 'textarea',
          ai_fillable=True,
          profile_role='intention',
          hubspot_property='what_do_you_want_non_dilutive_funding_to_accomplish_over_the_next_12_to_24_months___2_to_3_sentence', hubspot_object='contact'),
    Field('funding_target', '5_funding', 'Total non-dilutive funding you are aiming to raise in that period:', 'multi_select',
          options=('Under $250k', 'Between $250k - $1M', 'Between $1M - $5M', 'Between $5M - $10M', 'Over $10M'),
          option_values=('ugpLCgbkOg3QyzOZZUQmM', 'jP8PIvy7sKcBc2x4GEt-g', 'NuZEoZF8byosH_0GVe8Wd', 'jMxwZG2DFY16ASFST4Ypd', 'xTCovYYwTpkNsqRPtFDhB'),
          hubspot_property='total_non_dilutive_funding_you_are_aiming_to_raise_in_that_period_', hubspot_object='contact'),
    Field('earliest_start', '5_funding', 'Earliest you could start a funded project:', 'multi_select',
          options=('Within 3 Months', 'Between 3 - 6 Months', 'Between 6 - 12 Months', '12 Months +'),
          option_values=('atASAHj4pdI19N1mvTZpa', 'mHpOAzT2QUObqEDTmnW8C', 'Lltc6p14Jj5jt5Q59w1_N', 'bfkizG2a_ZaJaMyJJQ79A'),
          hubspot_property='earliest_you_could_start_a_funded_project_', hubspot_object='contact'),
    Field('identified_solicitation', '5_funding', 'Have you already identified a specific solicitation or funding agency?', 'textarea',
          ai_fillable=True,
          profile_role='intention',
          hubspot_property='have_you_already_identified_a_specific_solicitation_or_funding_agency_', hubspot_object='contact'),
    Field('current_funding', '5_company_funding', 'How is your company currently funded today?', 'multi_select',
          options=('Bootstrapped', 'Commercial Revenue', 'Outside Investors', 'Federal Funding', 'Debt Financing', 'Other'),
          option_values=('pI9SqpvF17Deo2CHFSB4V', '-MtdJCqjvpb2V8kKOycoY', 'cN-IoAAQhX2PJ-sBY-opn', 'OC892ZBHJMECb6gYvz9Tq', '8b75qM6m8BJ8nDWa3-tx6', 'o8kDPx-udXHtCQUbbBCvi'),
          ai_fillable=True,
          hubspot_property='how_is_your_company_currently_funded_today_', hubspot_object='contact'),
    Field('investor_types', '5_company_funding', 'If your company is investor-backed, what type of outside investment have you received?', 'multi_select',
          options=('Angel / Pre-Seed', 'Venture Capital', 'Private Equity', 'Strategic / Corporate Investment', 'Family Office', 'Other'),
          option_values=('qgQlonjDzJjcZh3bmQMFH', 'L-axbLXi1SkdgqWmUXxBL', '1HKKWKOB1eQngmWeAjaaV', 'domi17woTNjaHJP7EYyai', 'W2-uQt5W5ItYfil47dx4n', 'Ssexyh0kOF8ikJ4zmjDXw'),
          condition=INVESTOR_CONDITION,
          ai_fillable=True,
          hubspot_property='if_your_company_is_investor_backed__what_type_of_outside_investment_have_you_received_', hubspot_object='contact'),
    Field('primary_investors', '5_company_funding', 'Who are some of your primary investors?', 'textarea',
          condition=INVESTOR_CONDITION,
          ai_fillable=True,
          hubspot_property='who_are_some_of_your_primary_investors_', hubspot_object='contact'),
    Field('next_milestone', '5_company_funding', 'What is the next key milestone your company needs to achieve?', 'textarea',
          ai_fillable=True,
          profile_role='intention',
          hubspot_property='what_is_the_next_key_milestone_your_company_needs_to_achieve_', hubspot_object='contact'),
    Field('milestone_timing', '5_company_funding', 'When do you expect to reach this milestone?', 'multi_select',
          options=('Within 6 Months', 'Between 6 - 12 Months', 'Between 12- 24 Months', 'More than 24 Months'),
          option_values=('b-WTBZf-QGQANZ92ryIGr', 'w9ghCap_UVkC5xmAIgBo5', 'LRpnTTVnYWiwTuDqqXXrt', 'k7NfN4hHCuzBEbvpWcmhN'),
          ai_fillable=True,
          hubspot_property='when_do_you_expect_to_reach_this_milestone_', hubspot_object='contact'),
    Field('milestone_funding', '5_company_funding', 'Which best describes your ability to reach this milestone with your current funding?', 'multi_select',
          options=('We are currently funded and on track to reach it.', 'We can likely reach it with our current resources, but additional funding would accelerate us.', 'We need additional funding to reach it.', 'Not sure'),
          option_values=('z864vakCfsbiz_M9exwuu', 'jOIqaagobatrxmqqe5D-b', 'RVRwXpg1V7XTbhzJfdLV7', 'NRhNwuxYkFVNGA4GrEmco'),
          hubspot_property='which_best_describes_your_ability_to_reach_this_milestone_with_your_current_funding_', hubspot_object='contact'),
    Field('open_to_adjacent', '5_company_funding', 'Would you be open to pursuing R&D projects that are slightly different or adjacent to your primary project if doing so could create access to significant additional non-dilutive funding?', 'multi_select',
          options=('Yes', 'Possibly, depending on the project', 'No'),
          option_values=('Lrubth7i3ZGF1WcC98PUr', 'IRzZFyKgG-Dz1QSnDMYCY', 'v9Ne-rMopGV9-vbfuKkad'),
          hubspot_property='would_you_be_open_to_pursuing_r_amp_d_projects_that_are_slightly_different_or_adjacent_to_your_prim', hubspot_object='contact'),
    Field('adjacent_markets', '5_company_funding', 'If yes, what adjacent markets, applications, use cases, indications, or patient populations have you considered, even if they are not part of your current primary market or beachhead strategy?', 'textarea',
          condition=ADJACENT_CONDITION,
          ai_fillable=True,
          profile_role='intention',
          hubspot_property='if_yes__what_adjacent_markets__applications__use_cases__indications__or_patient_populations_have_yo', hubspot_object='contact'),
    Field('federal_traction', '5a_federal', 'Which best describes your current level of traction with the federal government?', 'multi_select',
          options=('We have not yet explored federal government applications or spoken with anyone in the federal government.', 'We believe the federal government may have an application for our technology, or others have told us the government could be interested, but we have not yet spoken directly with federal stakeholders.', 'We have spoken directly with federal stakeholders, but have not yet received clear confirmation that they want or need our solution.', 'We have spoken directly with federal stakeholders who have confirmed a need or expressed interest in our solution.', 'We are currently discussing a pilot, funded project, procurement, or other specific next step with the federal government.', 'We currently have or previously had a funded federal project or federal customer.'),
          option_values=('h5Rpr8d_3tSt2pq-TbHbJ', 'WhrQxyhG6CN-4X90pFZKx', 'dXydMJKL-3NwaZEcDFtfD', 'q7hgZFqZNl_n29ewc82KP', 'L3kfCwXzXKrTh-CchUzlD', 'V19LPLqsALLVoH35nqLML'),
          hubspot_property='which_best_describes_your_current_level_of_traction_with_the_federal_government_', hubspot_object='contact'),
    Field('agency_conversations', '5a_federal', 'Have you spoken with any program managers, contracting officers, or agency staff? If yes, which agency?', 'textarea',
          profile_role='intention',
          hubspot_property='have_you_spoken_with_any_program_managers__contracting_officers__or_agency_staff__if_yes__which_age', hubspot_object='contact'),
    Field('federal_funding_received', '5a_federal', 'Has your company previously received funding from the federal government?', 'multi_select',
          options=('No', 'SBIR/STTR', 'Other federal grant or cooperative agreement', 'Federal contract', 'OTA or prototype agreement', 'CRADA or other federal partnership', 'Other'),
          option_values=('FAZUw5Xdhs3Uqv8ads9SC', 'Ld16d4QEwjO3DXt_NCDc4', '4MC_tH5zSZdUDkOnyDmIt', 'XjxL5ZWbP3y1C2VEKK6tz', '4AoAI-odX-XVMIhNFz2in', 'TFynzW0rP9ylo3mAB76j8', '0NVbB9JoFxb07nQAPrYIM'),
          enrichable=True,
          profile_role='capability',
          hubspot_property='has_your_company_previously_received_funding_from_the_federal_government_', hubspot_object='contact'),
    Field('federal_funding_detail', '5a_federal', 'If yes, please briefly list the relevant agency, program, year, and approximate funding amount, if known.', 'textarea',
          condition=FEDERAL_CONDITION,
          enrichable=True,
          profile_role='capability',
          hubspot_property='if_yes__please_briefly_list_the_relevant_agency__program__year__and_approximate_funding_amount__if_', hubspot_object='contact'),
    Field('who_pays', '6_commercialization', 'Who pays for this once it exists?', 'multi_select',
          options=('Government Customers / Agencies', 'Commercial Customers (Businesses)', 'Individual Consumers', 'Not Yet Defined'),
          option_values=('hgr2gWrBWURztBJ5iRwBs', 'qQziOmFHrweILWF2RZnaj', '5iibBbwnpJB_zf-WYEFtc', 'Db7P7Xor6XXGftvR3Xv3o'),
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='who_pays_for_this_once_it_exists_', hubspot_object='contact'),
    Field('regulatory_pathway', '6_commercialization', 'Any regulatory pathway involved (FDA, FAA, EPA, other)?', 'textarea',
          ai_fillable=True,
          profile_role='capability',
          hubspot_property='any_regulatory_pathway_involved__fda__faa__epa__other__', hubspot_object='contact'),
    Field('short_term_goal', '6_commercialization', 'Top short-term business goal in one sentence', 'textarea',
          ai_fillable=True,
          profile_role='intention',
          hubspot_property='top_short_term_business_goal_in_one_sentence', hubspot_object='contact'),
    Field('uploads', '7_uploads', 'Upload files here:', 'file'),
]

FIELDS_BY_ID: dict[str, Field] = {f.id: f for f in FIELDS}
assert all(v in FIELDS_BY_ID[VERTICALS].options for v in HEALTH_VERTICALS)


def validate_schema(fields: list[Field] = FIELDS) -> list[str]:
    """Structural checks on the schema itself (run by the tests)."""
    errors: list[str] = []
    seen: set[str] = set()
    ids = {x.id for x in fields}
    for f in fields:
        if f.id in seen:
            errors.append(f'duplicate id {f.id}')
        seen.add(f.id)
        if f.section not in SECTION_IDS:
            errors.append(f'{f.id}: unknown section {f.section}')
        if f.type not in TYPES:
            errors.append(f'{f.id}: unknown type {f.type}')
        if f.profile_role not in PROFILE_ROLES:
            errors.append(f'{f.id}: unknown profile_role {f.profile_role}')
        if f.type in ('single_select', 'multi_select') and not f.options:
            errors.append(f'{f.id}: select without options')
        if f.option_values and len(f.option_values) != len(f.options):
            errors.append(f'{f.id}: option_values not parallel to options')
        for clause in f.condition or ():
            target = next((x for x in fields if x.id == clause.get('field')), None)
            if target is None:
                errors.append(f'{f.id}: condition on unknown field {clause.get("field")}')
            elif not set(clause['any_of']) <= set(target.options):
                errors.append(f'{f.id}: condition values not options of {target.id}')
    for core in (COMPANY_NAME, WEBSITE, CONTACT_EMAIL, CONTACT_FIRST, CONTACT_LAST,
                 TECH_DESCRIPTION, VERTICALS):
        if core not in ids:
            errors.append(f'core field {core} missing')
    return errors

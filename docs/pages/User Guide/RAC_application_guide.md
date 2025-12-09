---
title: "RAC application guide/en"
url: "https://docs.alliancecan.ca/wiki/RAC_application_guide/en"
category: "User Guide"
last_modified: "2025-11-03T20:57:51Z"
page_id: 29716
display_title: "RAC application guide"
---

Please read this guide carefully.

== Application processes and key dates ==

Application	Start             	Deadline
Resources for Research Groups (RRG)

* Including Fast Track applications
Research Platforms and Portals (RPP)

* Including RPP Annual Progress Reports	September 23, 2025	November 4, 2025, at 11:59 p.m. Eastern Standard Time

Extension of this deadline is not possible.

The Resource Allocation Competition (RAC) enables faculty members and their research groups to access compute, storage and cloud resources beyond what can be obtained via the Rapid Access Service (RAS). The RAC is overseen by the Resource Access Program Administrative Committee (RAPAC), which includes representatives from all regional partners and each of the national system host sites.

Important:

* If you are planning to request GPU and/or cloud resources, please attend one of the corresponding information sessions.
* If you are applying for the first time, we strongly encourage you to schedule a pre-submission consultation before October 31.
* You must use the most up-to-date version of the application template, which is provided in this guide. Failing to do so may result in the rejection of your application.
* Consult the evaluation criteria available in Appendix A and Appendix B of this document.

You may also wish to consult the RAC Frequently Asked Questions page and Technical glossary.

If you don't know which application process is best for your project, please email allocations@tech.alliancecan.ca.

=== Resources for Research Groups (RRG) ===
The RRG is a peer reviewed application process for projects whose primary purpose is to conduct research requiring compute, storage and cloud resources to meet their goals. However, projects primarily needing persistent instances in the cloud to provide a service through a platform or a portal should apply through the RPP application process instead.

Refer to the Appendices at the end of this Guide for the RRG Evaluation Criteria and scoring matrix.

==== RRG Fast Track Application Process ====
Users with an existing RRG award who meet the Fast Track eligibility are allowed to submit a lightweight progress report to renew their request for computational resources. Please read the Fast Track guidelines as this process includes important conditions and limitations that should be considered when deciding whether to use the Fast Track process or submit a new application.

Principal Investigators (PIs) with an active allocation awarded through the RRG process will receive an email in September indicating whether they are eligible to Fast Track or not. Fast Track submissions start on September 23 and are due November 4, 2025, at 11:59 p.m. Eastern Standard Time (extension of this deadline is not possible).

=== Research Platforms and Portals (RPP) ===
The RPP is a peer reviewed application process for projects whose primary purpose is to provide a service through scientific gateways that improve access to shared datasets, enhance existing online research tools and facilities, or advance national or international research collaborations. However, projects primarily needing compute resources in a cluster to conduct research should apply through the RRG process instead.

A research platform or portal is a set of community-developed tools, applications and data that are integrated via a gateway or a suite of applications—usually in a graphical user interface—that is further customized to meet the needs of a community of users associated with a specific discipline.

RPP projects typically involve cloud resources, usually through the development of a front-end gateway on persistent virtual machines, with possible backend compute either through cloud compute nodes or job-based submission to the national clusters. Many platforms and portals also include large databases.

Projects applying must:

# provide a service  to a larger research community via a set of cloud-based tools, applications and/or data, thus enabling them to access national computational resources via a common interface;
# be able to develop, operate and manage the proposed portal or platform with minimal support from the Federation.

Please refer to the Appendices at the end of this Guide for the RPP Evaluation Criteria and the scoring matrix.

Important: If your multi-year RPP award expires in 2026, you must submit a new RPP application this fall if your need for resources will continue for the 2026-2027 allocation cycle.

==== RPP Annual Progress Report ====
Allocations may be awarded over multiple years (maximum of three years), subject to an annual review and availability of resources.

Awarded multi-year projects do not need to submit a new application every year but are required to complete an annual progress report. The PI will be notified by email with instructions for the information required and the submission process. For more details, visit the RPP Progress Report page.

Submissions of the annual RPP Progress Report start September 23 and are due November 4, 2025.

= Eligibility =
To be eligible to submit a RAC application, the PI and all Co-PIs must:

* be a faculty member at a Canadian academic institution: and
* have an active Alliance account in the CCDB with an Academic PI role (Faculty, Adjunct Faculty or Librarian).

Eligible candidates may

* apply as PI for only one RRG application at a time (either through the full application or the Fast Track process) but can be involved in multiple RRG submissions as a co-PI or a collaborator; and
* be the PI for one RPP application per competition round and be involved in other RPP applications as a co-PI.

Keep in mind:
* An eligible PI must apply with their primary, most up-to-date position. If the PI recently moved to a different institution and has not yet applied for a new faculty role on the CCDB portal, please do so before submitting your RAC application.
* If the PI has more than one active faculty role in the CCDB portal, please make sure that the most up-to-date position is set as the primary role. To do so, go to the Home page in the CCDB portal to see which of your roles is currently set as primary and, if needed, click on the Make this role primary button next to the new role that you want to set as primary.

Failing to do the above may cause problems if your application is successful.

Important: “Departmental" applications will not be accepted (i.e., applications submitted on behalf of a group of PIs that may be from the same department but are not collaborating in a common research project with clear goals and outcomes).

=== Co-PIs and Collaborators ===
In the context of this competition, a Co-PI is any Canadian faculty with an Alliance Academic PI role that is actively involved in the computational project. The PI and any Co-PI must upload an updated CCV with the application (see CCV Requirement). The role and involvement of any Co-PI listed in the application must be justified.

A Collaborator is any other person actively participating in the project that cannot be listed as a co-PI based on the definition provide before. International investigators or colleagues without an Alliance account can be listed as collaborators in the Resource Management section of the application document.
=== Minimum Amount of Resources Eligible for RAC ===
If your need for computational resources can be met with what will be available via the Rapid Access Service, do not submit a RAC application.

Note that /HOME storage and /SCRATCH storage are not allocated through the RAC. If you don't understand some of the terms listed above, please visit the Technical glossary.

Category	Resource                                 	Minimum	Important information
HPC     	CPU (core-years or core-equivalent-years)	200    	A CPU request must be equal to or greater than 200 core-years or core-equivalent-years in each of the clusters where these resources are needed.
If you require 200 core-years or less but high memory per core, please use the following formula to calculate the core-equivalent-years to know whether you should submit an application.

core-equivalent-years = MAX(cores, mem requested / 4GB)

Mem requested = CY * mem per core requested

The resulting calculation must be greater than 200 for the request to be eligible for a RAC application.

For more information about compute allocations, core-equivalent-years, and scheduler priority, please visit the Allocation and compute scheduling page.
HPC     	GPU (RGU-years)                          	25     	A GPU request must be equal to or greater than 25 reference GPU units (RGU) years in each of the clusters where you need these resources.
RGU is a unit measuring the amount of GPU resources that are used. It represents the cost of utilizing a particular GPU model, whose RGU value varies based on performance. For example:

1 GPU A100-40GB = 4.0 RGU

1 GPU H100_2g.20GB = 3.5 RGU

1 GPU H100-80GB = 12.2 RGU

When requesting GPU resources in the online form, you will have to specify which GPU model you intend to use. We are "charging" GPU usage for different GPU models to account for the wide range of performance shown by the different models of GPU available. Note that our resources are provided for free to Canadian researchers; however, when determining who is to be scheduled next, we need to evaluate how many resources a group has used compared to their share. This allows us to prioritize groups that have not used their share. For more information about the GPU models available and what they will be charged for in RGU, visit this page.
HPC     	Project storage (TB)                     	41     	This limit applies to any General Purpose cluster (Fir, Nibi, Rorqual and Narval). Contrary to CPU and GPU, where the minimum amount is required for each cluster, the minimum of 41 TB is the sum of all the project storage needed across all clusters.
Important:

* If you need project and nearline storage, you can request both resources as long as one of the requests is equal to or greater than the minimum specified in this table. In that case, the online form must include the project and nearline the storage  resources needed.
* This minimum amount does not apply to the Trillium cluster; any amount of project storage needed on Trillium must be requested through the RAC.
HPC     	Nearline storage (TB)                    	101    	This limit applies to any General Purpose cluster (Fir, Nibi, Rorqual and Narval). Contrary to CPU and GPU, where the minimum amount is required for each cluster, the minimum of 101 TB is the sum of all the nearline storage needed across all clusters.
Important:

* If you need project and nearline storage, you can request both resources as long as one of the requests is equal to or greater than the minimum specified in this table. In that case, the online form must include the nearline and project storage resources needed.
* This minimum amount does not apply to the HPSS cluster; any amount of project storage needed on HPSS must be requested through the RAC.
HPC     	dCache storage (TB)                      	0      	Any amount of dCache storage must be requested.
Cloud   	Compute Cloud VCPU (VCPU-years)          	81     	As long as one of the resources needed in a particular cloud site is equal to or greater than the minimum specified in this table, the request is eligible. In that case, the online form must include all the cloud resources needed on that site, even if the amounts requested for some resources are below the minimum listed here.
Cloud   	Compute Cloud VGPU (RGU-years)           	3.6    	As long as one of the resources needed in a particular cloud site is equal to or greater than the minimum specified in this table, the request is eligible. In that case, the online form must include all the cloud resources needed on that site, even if the amounts requested for some resources are below the minimum listed here.
Cloud   	Persistent Cloud VCPU (VCPU-years)       	26     	As long as one of the resources needed in a particular cloud site is equal to or greater than the minimum specified in this table, the request is eligible. In that case, the online form must include all the cloud resources needed on that site, even if the amounts requested for some resources are below the minimum listed here.
Cloud   	Volume and Snapshot Storage (TB)         	11     	As long as one of the resources needed in a particular cloud site is equal to or greater than the minimum specified in this table, the request is eligible. In that case, the online form must include all the cloud resources needed on that site, even if the amounts requested for some resources are below the minimum listed here.
Cloud   	Shared Filesystem Storage (TB)           	11     	As long as one of the resources needed in a particular cloud site is equal to or greater than the minimum specified in this table, the request is eligible. In that case, the online form must include all the cloud resources needed on that site, even if the amounts requested for some resources are below the minimum listed here.
Cloud   	Object Storage (TB)                      	11     	As long as one of the resources needed in a particular cloud site is equal to or greater than the minimum specified in this table, the request is eligible. In that case, the online form must include all the cloud resources needed on that site, even if the amounts requested for some resources are below the minimum listed here.

=== Requesting Resources ===
It is extremely important that the online application form includes a request for every resource (CPU, GPU, storage or cloud) that your project needs on each cluster. There should be NO DISCREPANCIES between the amount of resources requested in your attached Application Document and what you request in the online form. In case of discrepancy, what was requested in the online form will prevail.

Please visit the Available Resources page for a list of systems available for this competition and to understand how resources must be requested in the online form.

Example of acceptable resource requests:

* If you need 180 core-years (or core-equivalent-years) and 50 TB of project storage on Fir, your RAC application should only request 50 TB of storage on that cluster. Do not request CPU in your application as you may be able to get the CPU that you need without a RAC allocation.
* If you need 500 core-years, 20 TB of project storage and 80 TB of nearline storage on Narval, your RAC application should only request 500 core-years on that cluster, as your storage needs can be met with what is available via RAS.
* If you need 100 core-years with 10 TB of project storage on Trillium and 50 TB of nearline storage on HPSS, then you should only request the 10 TB and 50 TB of storage on those clusters, respectively, as you may be able to get the CPU that you need without a RAC allocation.
* If you need 400 core-years, 30 RGU-years and 200 TB of project storage on Nibi, then you must request all these resources in your RAC application as they are all above the minimum eligible amounts specified in the table.
* If you need 150 core-years on Rorqual with different memory requirements: 100 cores with 16 GB per core (which results in 400 core-equivalent-years) and 50 cores with 4 GB per core (50 core-equivalent-years), then you should make two separate CPU requests on Rorqual to account for the different memory requirements. This request is acceptable because the total core-equivalent-years needed on a single cluster is greater than the minimum amount of CPU specified in the table. Another acceptable option would be to average out the total memory per core needed and just make one request for 150 cores with 12 GB per core.
* If you need 45 TB of project storage and you want to split it between Rorqual and Narval as follows: 20 TB on Rorqual and 25 TB on Narval, then this request is eligible since the total amount of project storage needed exceeds 41 TB.
* If you need a total of 100 TB of storage, split as follows: 50 TB of project storage and 50 TB of nearline on Fir, then this request is eligible because even though the nearline amount is below the minimum (101 TB), the project storage request is eligible. Therefore, you should include both requests in your application.
* If you need 100 VCPU of compute cloud resources on Arbutus with 5 TB of Volume and snapshot (TB) and 5 TB of object storage, then you should request all these resources in the online form. Since the amount of VCPUs needed on Arbutus exceeds the minimum required to apply, the storage resources, must be included in the same request even if they are less than 11 TB.

Example of resource requests that will be rejected:

* You need 250 core-years in total but you want to split the CPU request between Fir and Nibi as 100 cores with 4 GB per core on Fir and 150 cores with 4 GB per core on Nibi: this request will NOT be accepted because you may be able to get the CPU that you need without a RAC allocation. If your application only includes a CPU request in this way (that is, it does not include any additional storage, GPU or cloud resources), it will be disqualified.
* If you need 20 reference GPU units in total but you want to split the GPU request between Narval and Rorqual with 10 RGU and 10 RGU, respectively, this request will not be accepted.
* If you need 25 VCPU of persistent cloud resources on Arbutus with 10 TB of Volume and snapshot storage (TB), then you should not apply for RAC as these resources are available via RAS.

= Application document =
Applicants are required to use the template provided below for the corresponding application process. Always download the latest version of the applicable document(s) as changes in the instructions may happen from year to year, so it is important that your application uses the most up-to-date version of the template.

Important:

* Applications exceeding 10 pages (before References) will not be accepted.
* When applicable, References must be submitted as a separate document (if the total number of pages of the entire application, including References, is 10 pages or less, then submitting everything in just one document is acceptable).
* These documents must be submitted in PDF format .

Download the template required for the application process that you intend to apply to

Application process           	Template format
Resources for Research Groups 	- Word or OpenDocument* (English, French)

- LaTeX (English, French)
Research Platforms and Portals	- Word or OpenDocument* (English, French)

- LaTeX (English, French)

*This link will open a non-editable Google document (please do not request permission to edit this Google document). Once there, go to File → Download in the main menu and select Microsoft Word or OpenDocument Format to download the template; then save it and start working in the downloaded document.

= Submission Procedures and Deadlines =
Proposals must be submitted electronically through the CCDB portal no later than November 4, 2025, at 11:59 p.m. Eastern Standard Time. The PI is responsible for ensuring that the application is complete, with all additional documentation uploaded, and that there is no discrepancy between the application document and the online form.

Following the review process, applicants will be informed regarding the status of their applications via email in spring 2026.

=== Pre-Submission Consultations ===
If you are applying for the first time to this competition, we strongly encourage you to consult with us prior to submitting your application. Consultations should happen prior to October 31, 2025, to allow adequate time for support by our technical staff.

The goal of the consultation is to

* determine whether the resources needed for your project justify submitting a RAC application;
* verify the project's eligibility to the right application process (RPP vs RRG);
* provide technical assistance with the calculation of the resources needed.

To schedule a consultation, please send an email to allocations@tech.alliancecan.ca or contact your regional support team.

=== Online Application Form ===
All RAC applications are submitted online via the CCDB portal. Users must log in using an existing Alliance account or register for a new one.

=== CCV Requirement ===
An up-to-date Canadian Common CV (CCV) is required for the peer review process.

The PI and all Co-PIs are required to:

* Upload an up-to-date CCV with any RAC application.
* Report publications enabled by your use of resources provided by the Federation.

Failing to provide an updated CCV will negatively impact the overall score of the submitted application.

The PI can view in the online application form the date in which any CCV linked to the application was last uploaded and whether any action is required.

Co-PIs can update their CCVs on CCDB by clicking on Update CCV in the Resource Applications page, or by going to My Account → View Reporting. Once Co-PIs have updated their CCV, the status of the CCV will be automatically updated in the RAC online application form (PIs can also share this link https://ccdb.alliancecan.ca/reporting with their co-PIs and ask them to update their CCV).

Please plan ahead to avoid delays or the risk of missing the submission deadline due to this requirement. Deadline extensions due to missing CCVs are not possible.

Please carefully read the CCV Submission Guide for further instructions.

= Assessment Process =

=== Guiding Principles ===
The RAC is guided by the following principles:

* All applications are given fair consideration through both a peer review and a technical review;
* Resources are awarded based on the appropriateness of the computational resources requested to achieve the project's objectives and the likelihood that these resources will be efficiently used, rather than on the evaluation of a complete research program; and
* the challenges arising from the shortage of resources and other constraints within the system are shared among all applicants.

=== Technical Review ===
The technical review is conducted by technical experts of the Federation who:

* ensure the appropriate system is requested by the PI and the required software is available;
* evaluate application efficiency and scalability;
* identify groups that may need help with application and workflow optimization;
* identify discrepancies between the resource request included in the online form and the attached application document;
* identify special software requirements;
* provide a technical opinion on the reasonability of the request.

Technical reviewers are required to sign a Non-Disclosure Agreement prior to accessing any RAC application.

During the technical review process, staff may require additional information from applicants and will engage them directly. In order to ensure an application can progress beyond the technical review, applicants are expected to respond to requests from the technical team within 48 hours.

=== Peer Review ===
All applications submitted to the RAC are peer reviewed and scored. Peer reviewers are required to accept the Alliance's Non-Disclosure Agreement and Conflict of Interest Policy prior to accessing any application.

The resulting score is based on the following:

* the feasibility of the computational research project based on what research will be done and what outcomes will be delivered, and not on why it is important;
* the appropriateness of the resources requested to achieve the objectives of the computational research project based on the technical justification provided; and
* the likelihood that the resources requested will be efficiently used.

Applications will be reviewed in one of the committees below. PIs can select a peer review committee of their choosing; however, applications can be moved to a different committee following consultation with the committee Chairs.

* Astronomy, Astrophysics and Cosmology
* Bioinformatics
* Chemistry, Biochemistry and Biophysics
* Computer Sciences and Mathematics
* Engineering
* Environmental and Earth Sciences
* Humanities and Social Sciences
* Nano, Materials and Condensed Matter
* Neurosciences, Medical Imaging and Medical Physics
* Subatomic Physics, Nuclear Physics and Space Physics

=== Resource scaling ===
Resources provided by the Federation are limited, and for this reason requests for allocations are scaled every year based on the overall score of the application and the supply and demand.

A scaling function is applied to compute requests to provide a means by which decisions on RAC allocations in a context of insufficient capacity can be made. Visit the RAC Annual Reports page for more details about the scaling function and other stats from previous years.

= Questions and Additional Information =
For any questions, please contact allocations@tech.alliancecan.ca. You may also wish to consult our Frequently Asked Questions page or the Technical glossary.

= APPENDIX A: Resources for Research Groups Evaluation Criteria =
RRG applications are evaluated against the two following criteria: Research Methods and Resource Management and Computational Expertise.

=== Research Methods (70%) ===
This criterion evaluates the methods proposed to achieve the objectives of the computational project and the appropriateness of the resources requested. It focuses more on assessing what research will be done with the resources requested and on the technical justification provided, than on why the research is important.

Considerations for the evaluation of this criterion include the following:

Research Outline

* The research problem is clearly presented.

* The overall goal and objectives of the project are well-defined and clear, and they state what the computational project is ultimately expected to achieve.

Expected Outcomes

* The application presents anticipated outcomes and indicates the means by which these will be measured.
* The proposed computational project outputs (i.e., the anticipated results of the project) and impact are clearly described, are aligned to the objectives, are of relevance and are realistic and attainable.
* The proposed computational research project is likely to lead to advances in the research area.

Progress Over the Past Year

* The application shows achievements, outcomes and/or evidence of progress resulting from the utilization of resources provided by the Federation over the past year.

Computational Methods

* The application describes appropriate tools, methods and approaches for addressing the research objectives. These methodologies may be community codes or models, data analysis methods, algorithmic formulations expressed in user-developed scripts or tools, as well as trials or test implementations.

Resource Request Justification

* When applicable, a justification for low utilization (or lack thereof) of an existing allocation is provided and deemed reasonable.
* The amount of resources requested is deemed appropriate to achieve the project objectives, and the technical justification provided is excellent.
* The application describes necessary and sufficient computational experiments to answer the research questions posed.

=== Resource Management and Computational Expertise (30%) ===
This criterion evaluates the capacity of the research team as a whole to manage the project and make efficient use of the resources requested. It also assesses the overall feasibility of the computational project based on the research and computational expertise of the team.

The team includes the Principal Investigator (PI) and, if applicable, Co-PIs and any Highly Qualified Personnel (HQP) actively participating in the computational project.

HQP includes all research personnel involved in the applicant’s computational project, whether from academia, government or industry. The number of HQP using resources provided by the Federation directly is expected to be appropriate to the scope of the project.

The nature, breadth and depth of the applicants’ (PI and, if applicable, Co-PIs) experiences and contributions should be assessed in the context of their career stages. Committee members must not impart, refer to or consider information about the applicants that does not appear in the application and the provided Canadian Common CV (CCV).

It is not mandatory that an application includes Co-PIs or HQP to obtain the full Resource Management score. However, the proposed research must be achievable by the listed team members, particularly if it is only one PI and/or if there is no funding available. It is expected that applications asking for large amounts of resources will have funding to justify the request.

The level of detail needed to get a high score for the Resource Management criterion is a function of team size and resource ask.

Considerations of this review criterion include the following:

Funding

* Funding is available for the research project to justify the request for computational resources; when funding is not available, a reasonable explanation for how the compute resources will be utilized is provided.

Computational Expertise of the Team

* The team shows sufficient computational expertise or a training plan to make effective and efficient use of the computational resources requested.

Management Strategy

* Information about each team member (PI, Co-PIs and HQP, when applicable) and their requested computational resources are clearly described.
* The roles and responsibilities of the PI and Co-PIs, if applicable, are clearly described with respect to making efficient use of the resources requested and are linked to the objectives of the computational project (Co-PIs contribute to the Resource Management score in accordance with their involvement).

* The team demonstrates the combined expertise and experience needed to execute the computational project, i.e., deliver the proposed outputs as well as achieve the proposed contribution(s).

= APPENDIX B: Research Platforms and Portals Evaluation Criteria =
RPP applications are evaluated based on the following two criteria: Project Justification and Resource Management and Expertise of the Team.

=== Project Justification (50%) ===
Considerations for the evaluation of this criterion include the following:

Project Description, Objectives and Impact

* The problem or need that the research platform/portal will address is clearly presented.
* The objectives and goals of the platform/portal clearly described.

Use of the Platform/Portal

* The added value from the creation or maintenance of the platform or portal for the targeted communities is clearly explained.
* If applicable, the level of interaction between Canadian and international research groups is clearly described.

Expected Outcomes

* The application presents a clear timeline for the delivery of the anticipated outcomes over the entire duration of the requested resources and indicates the means by which these outcomes will be measured.

Progress over the Past Year

* The application shows achievements, outcomes and/or evidence of progress resulting from the utilization of resources provided by the Federation over the past year.
* When applicable, a justification for low utilization (or lack thereof) of an existing allocation is provided and deemed reasonable.

Resource Request Justification

* The amount of resources requested is deemed appropriate to achieve the project objectives, and the technical justification provided is solid.

=== Resource Management and Expertise of the Team (50%) ===
This criterion evaluates the capacity of the research team as a whole to manage, develop and operate the platform/portal, and the ability of the team to make efficient use of the resources requested. It also assesses the overall feasibility of the project based on the expertise of the team.

The team includes the Principal Investigator (PI) and, if applicable, Co-PIs and any Highly Qualified Personnel (HQP) actively participating in the project.

HQP includes all research personnel involved in the project, whether from academia, government, or industry. The number of HQP directly involved in the management, development and operation of the platform/portal is expected to be appropriate to the scope of the project.

It is expected that applications asking for large amounts of resources will have sufficient funding to develop, manage and operate the platform/portal.

Considerations of this criterion include:

Funding

* Funding is available to support the management, development and operation of the platform/portal and to justify the request for computational resources; when funding is not available, a reasonable explanation for how the platform/portal will be managed, operated and developed and how the compute resources will be utilized is provided.

Team Configuration and Expertise

* The team assembled to manage, develop and operate the platform has the right combination of skills (where positions are not yet filled, a description of the position has been included).

Management Strategy

* The proposed management of the resources is well defined and will provide broad access to the research community.
* The process for resource access is well defined, and a credible strategy to maintain or increase the community accessing resources is described.
* The proposed methods and technologies are suitable and well justified for the services to be provided by the platform.
* The approach to sharing data sets across the platform or portal is well described, and potential accessibility issues are properly addressed.

= APPENDIX C: Scoring matrix =
RAC applications are scored based on a 5-point scale as shown in the table below. Applications with a score of 2.0 or lower are considered unsuccessful and will not be awarded.

Descriptor  	Range    	Definitions
Excellent   	4.1 - 5.0	The application excels in all relevant aspects of the review criteria. Any shortcomings are minimal.
Good        	3.1 - 4.0	The application excels in most relevant aspects of the review criteria and reasonably addresses all others. Certain improvements are possible.
Fair        	2.1 - 3.0	The application excels in some relevant aspects of the review criteria. Relevant aspects could be better addressed and/or need to be revised or improved.
Poor        	1.1 - 2.0	The application broadly addresses relevant aspects of the review criteria. Relevant aspects of the review criteria are unclear, are missing, or require major revisions or improvements .
Insufficient	0.1 - 1.0	The application fails to provide convincing information, has serious inherent flaws or gaps and/or relevant aspects of the review criteria are missing. Extensive revision will be required.

= APPENDIX D: Non-disclosure Agreement and Conflict of Interest Policy for Peer Reviewers =
Definitions:

* Scientific Review Committee: panel of experts in multiple disciplines that volunteer to peer review applications submitted to the Resource Allocation Competition (RAC).
* Applicant: The Principal Investigator (PI) and any co-PI listed in an application submitted to the Resource Allocation Competition (RAC).

By signing this agreement, you are accepting to abide by the policy outlined below.

The undersigned agrees to treat as strictly confidential all information received for the purpose of evaluating resource applications, as well as all unpublished material from the documents during the review process, together with all deliberations, comments, scores and recommendations of the peer review committees. This information must not be used for any purpose beyond that for which it was originally intended.

Conflict of Interest Policy

Reviewers  must follow the guidelines below regarding conflicts of interest.

Reviewers must disclose, as early as possible, any conflict of interests with a RAC application to which they are directly or indirectly associated. These guidelines cannot foresee all possible situations and the Alliance must rely on the judgment of the reviewers to disclose these conflicts of interests.

Reviewers are in direct conflict if they:

* submitted an application that will be reviewed in the committee they are serving (in this case, they are in conflict with their own application only);
* are from the same university department as the applicant;
* have been a research supervisor or graduate student of the applicant within the past five (5) years;
* have collaborated or published with the applicant within the past two (2) years or have plans to collaborate or publish in the immediate future;
* are providing direct or indirect support for the application;
* are a relative or close friend, or have a personal relationship with the applicants;
* have had scientific or personal differences with the applicant that have been expressed publicly;
* are in a position to gain or lose financially from the outcome of the application (e.g., hold stock in the company of an industry partner or a competitor), or for some other reason feel that they cannot provide an objective review of the application.

Committee members are in indirect conflict if they:

* are from the same immediate institution* or company as the applicant and interact with the applicant in the course of their duties at the institution or company;
* have other reasons not to review an application.

*A reviewer is not in conflict with an application if they are from the same institution as the applicant but do not know or interact with the latter.

Difficult cases should be brought to the administrators of the RAC process, who have the authority  to rule.

All peer reviewers must read and agree to abide by this conflict of interest policy prior to viewing any application information.

= APPENDIX E: Confidentiality of Information =
The Alliance safeguards the information it receives from applicants. All reviewers are required to sign a Non-Disclosure Agreement and Conflict of Interest Policy. They are instructed to keep all proposal information confidential and to use it only for review purposes. All proposals are available for review by all reviewers and the Resource Access Program Administrative Committee (RAPAC).

Use of Personal Information

Any personal information collected by the Alliance is used only to review applications. Such information may be shared with relevant officials in the corresponding regional partner, as well as with the research institution of these officials.

Public Information

The Alliance will make public the following information about successful applications:

* PI name
* Institution
* Department
* Research area
* Project title
* Project summary
* Summary of resources awarded
* Application process
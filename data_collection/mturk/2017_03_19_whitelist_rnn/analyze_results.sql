drop table if exists mturk cascade;
create table mturk (
    HITId varchar,
    HITTypeId varchar,
    Title varchar,
    Description varchar,
    Keywords varchar,
    Reward varchar,
    CreationTime varchar,
    MaxAssignments integer,
    RequesterAnnotation varchar,
    AssignmentDurationInSeconds integer,
    AutoApprovalDelayInSeconds integer,
    Expiration varchar,
    NumberOfSimilarHITs varchar,
    LifetimeInSeconds varchar,
    AssignmentID varchar,
    WorkerId varchar,
    AssignmentStatus varchar,
    AcceptTime varchar,
    SubmitTime timestamp,
    AutoApprovalTime varchar,
    ApprovalTime varchar,
    RejectionTime varchar,
    RequesterFeedback varchar,
    WorkTimeInSeconds integer,
    LifetimeApprovalRate varchar,
    Last30DaysApprovalRate varchar,
    Last7DaysApprovalRate varchar,
    Input_image_A_url varchar,
    Input_image_B_url varchar,
    Answer_choice varchar
);

\copy mturk from '/home/jonathan/Documents/2017/ece496/gif-gan/data_collection/mturk/2017_03_19_whitelist_rnn/Batch_2731339_batch_results.csv' delimiter ','  csv;

-- Overall stats
select
    avg(WorkTimeInSeconds) average_completion_time,
    count(case when Answer_choice='optionA' then 1 end)*100.0/
    count(*) percentage_prefer_a,
    count(distinct WorkerId) num_workers
from mturk;

-- Worker stats
select
    WorkerId,
    count(case when Answer_choice='optionA' then 1 end) num_a,
    count(case when Answer_choice='optionB' then 1 end) num_b,
    count(case when Answer_choice='optionA' then 1 end)*100.0/
    count(*) percentage_prefer_a,
    count(*) num_responses
from mturk
group by WorkerId
order by percentage_prefer_a desc, num_responses desc;

-- Pair stats (super noisy)
-- select
--     HITId,
--     count(case when Answer_choice='optionA' then 1 end) num_a,
--     count(case when Answer_choice='optionB' then 1 end) num_b,
--     count(case when Answer_choice='optionA' then 1 end)*100.0/
--     count(*) percentage_prefer_a,
--     count(*) num_responses
-- from mturk
-- group by HITId;

-- Best pairs for option A
select
    HITId,
    count(case when Answer_choice='optionA' then 1 end) num_a,
    count(case when Answer_choice='optionB' then 1 end) num_b,
    count(case when Answer_choice='optionA' then 1 end)*100.0/
    count(*) percentage_prefer_a,
    count(*) num_responses,
    max(Input_image_A_url) image_a,
    max(Input_image_B_url) image_b
from mturk
group by HITId
order by percentage_prefer_a desc
limit 10;

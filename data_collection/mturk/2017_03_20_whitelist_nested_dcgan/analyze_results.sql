drop table if exists mturk_swap cascade;
create table mturk_swap (
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
    Input_swap integer,
    Answer_choice varchar
);

-- Change the file path to whereever your results csv is, has to be absolute path too
-- Also don't forget to delete the first line of the csv
\copy mturk_swap from '/home/jonathan/Documents/2017/ece496/gif-gan/data_collection/mturk/2017_03_20_whitelist_nested_dcgan/Batch_2733070_batch_results.csv' delimiter ','  csv;

-- Unswap the swapped pairs
drop view if exists mturk cascade;
create view mturk as
select 
    *,
    case
        when Answer_choice='optionA' and Input_swap=0 then 'A'
        when Answer_choice='optionB' and Input_swap=1 then 'A'
        when Answer_choice='optionB' and INput_swap=0 then 'B'
        when Answer_choice='optionA' and INput_swap=1 then 'B'
    end as Answer
from mturk_swap;

-- Overall stats
select
    avg(WorkTimeInSeconds) average_completion_time,
    count(case when Answer='A' then 1 end)*100.0/
    count(*) percentage_prefer_a,
    count(distinct WorkerId) num_workers
from mturk;

-- Worker stats
select
    WorkerId,
    count(case when Answer='A' then 1 end) num_a,
    count(case when Answer='B' then 1 end) num_b,
    count(case when Answer='A' then 1 end)*100.0/
    count(*) percentage_prefer_a,
    count(*) num_responses
from mturk
group by WorkerId
order by percentage_prefer_a desc, num_responses desc;

-- Pair stats (super noisy)
-- select
--     HITId,
--     count(case when Answer='A' then 1 end) num_a,
--     count(case when Answer='B' then 1 end) num_b,
--     count(case when Answer='A' then 1 end)*100.0/
--     count(*) percentage_prefer_a,
--     count(*) num_responses
-- from mturk
-- group by HITId;

-- Best pairs for option A
select
    HITId,
    count(case when Answer='A' then 1 end) num_a,
    count(case when Answer='B' then 1 end) num_b,
    count(case when Answer='A' then 1 end)*100.0/
    count(*) percentage_prefer_a,
    count(*) num_responses,
    max(Input_image_A_url) image_a,
    max(Input_image_B_url) image_b
from mturk
group by HITId
order by percentage_prefer_a desc
limit 10;

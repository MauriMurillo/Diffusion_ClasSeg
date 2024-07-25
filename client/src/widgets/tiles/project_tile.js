import './tilesStyles.css';

function Tile(props) {
    // console.log(project)
    let {project, onClick} = props;
    let {name, type, preprocessed_available, raw_available, experiment_count} = project;

    let description =
        `Type: ${type}\nPreprocessed: ${preprocessed_available}\nRaw Available: ${raw_available}\nExperiments: ${experiment_count}`;
    return (
        <div className="ProjectTile">
            <h3>{name}</h3>
            <p>{`Dataset Type: ${type}`}</p>
            <p>{`Preprocessed: ${preprocessed_available}`}</p>
            <p>{`Raw Available: ${raw_available}`}</p>
            <p>{`Experiments: ${experiment_count}`}</p>
            <button onClick={() => onClick()}>Inspect</button>
        </div>
    )
}

export default Tile;